from __future__ import annotations

import random

from octocortex.arms.memory import MemoryArm
from octocortex.arms.navigation import MOVES, NavigationArm
from octocortex.arms.prediction import PredictionArm
from octocortex.arms.vision import VisionArm
from octocortex.core.event_bus import SparseEventBus
from octocortex.core.capabilities import CapabilityCode, CapabilityRouter
from octocortex.core.models import ActionCode
from octocortex.core.octoir import ConceptCode, SemanticPacket, UnitCode
from octocortex.core.workspace import GlobalWorkspace
from octocortex.learning.adapter import SparseSemanticAdapter
from octocortex.learning.backup_planner import DistilledBackupPlanner
from octocortex.simulation.worlds import DEFAULT_WORLD, TRAIN_WORLDS, WorldSpec


class OctoSimulation:
    def __init__(
        self,
        seed: int = 7,
        sensor_noise: float = 0.0,
        disabled_units: frozenset[UnitCode] = frozenset(),
        world: WorldSpec = DEFAULT_WORLD,
        backup_training_worlds: tuple[WorldSpec, ...] = TRAIN_WORLDS,
        partial_observability: bool = False,
        enable_prediction: bool = True,
        sensor_radius: int = 1,
        prediction_stale_after: int | None = None,
    ) -> None:
        if not 0.0 <= sensor_noise <= 1.0:
            raise ValueError("sensor_noise must be between 0 and 1")
        self.seed = seed
        self.sensor_noise = sensor_noise
        self.disabled_units = frozenset(disabled_units)
        self.world = world
        self.backup_training_worlds = backup_training_worlds
        self.partial_observability = partial_observability
        self.enable_prediction = enable_prediction
        self.sensor_radius = sensor_radius
        self.prediction_stale_after = prediction_stale_after
        self.random = random.Random(seed)
        self.reset()

    def reset(self) -> dict:
        # Reset the cognitive state as well as the body/world state so repeated
        # experiments remain comparable from the browser.
        self.random.seed(self.seed)
        self.bus = SparseEventBus(threshold=0.5)
        self.workspace = GlobalWorkspace()
        self.capabilities = CapabilityRouter()
        self.adapter = SparseSemanticAdapter()
        self.vision = VisionArm(self.adapter)
        self.memory = MemoryArm(self.adapter)
        self.navigation = NavigationArm(self.adapter)
        self.prediction = PredictionArm(self.adapter, stale_after=self.prediction_stale_after)
        self.tick = 0
        self.agent = self.world.start
        self.visited_cells = {self.agent}
        self.goal = self.world.goal
        self.obstacles = set(self.world.obstacles_at(0))
        self.backup_planners = {
            UnitCode.MEMORY: DistilledBackupPlanner(UnitCode.MEMORY, seed=41),
            UnitCode.PERCEPTION: DistilledBackupPlanner(UnitCode.PERCEPTION, seed=43),
        }
        training_states = tuple(world.state() for world in self.backup_training_worlds)
        for planner in self.backup_planners.values():
            planner.distill_many(training_states)
        self.collisions = 0
        self.done = False
        self.last_decision = None
        return self.snapshot()

    def _state(self) -> dict:
        return {
            "size": list(self.world.size),
            "agent": list(self.agent),
            "goal": list(self.goal),
            "obstacles": [list(cell) for cell in sorted(self.obstacles)],
        }

    def _perceived_state(self) -> dict:
        state = self._state()
        if not self.sensor_noise:
            return state
        perceived = {tuple(cell) for cell in state["obstacles"]}
        x, y = self.agent
        width, height = state["size"]
        for dx, dy in MOVES.values():
            cell = (x + dx, y + dy)
            if 0 <= cell[0] < width and 0 <= cell[1] < height and self.random.random() < self.sensor_noise:
                perceived.symmetric_difference_update({cell})
        return {**state, "obstacles": [list(cell) for cell in sorted(perceived)]}

    def _local_observation(self) -> dict:
        width, height = self.world.size
        x, y = self.agent
        visible = []
        perceived_obstacles = set()
        for cy in range(max(0, y - self.sensor_radius), min(height, y + self.sensor_radius + 1)):
            for cx in range(max(0, x - self.sensor_radius), min(width, x + self.sensor_radius + 1)):
                occupied = (cx, cy) in self.obstacles
                if self.sensor_noise and self.random.random() < self.sensor_noise:
                    occupied = not occupied
                visible.append([cx, cy, occupied])
                if occupied:
                    perceived_obstacles.add((cx, cy))
        return {
            "size": [width, height],
            "agent": list(self.agent),
            "goal": list(self.goal),
            "obstacles": [list(cell) for cell in sorted(perceived_obstacles)],
            "visible": visible,
            "observed_cells": len(visible),
        }

    def step(self) -> dict:
        if self.done:
            return self.snapshot()
        self.tick += 1
        self.obstacles = set(self.world.obstacles_at(self.tick))
        state = self._state()
        sensor_state = self._local_observation() if self.partial_observability else self._perceived_state()
        prediction_events = []
        if self.partial_observability and self.enable_prediction and UnitCode.PREDICTION not in self.disabled_units:
            prediction_events = self.prediction.observe(sensor_state, self.tick)
            perceived_state = self.prediction.belief_state(self.agent, self.goal)
        else:
            perceived_state = sensor_state

        available = {UnitCode.PLANNING, UnitCode.MEMORY, UnitCode.PERCEPTION} - set(self.disabled_units)
        planning_lease = self.capabilities.resolve(CapabilityCode.PLAN_ROUTE, available)
        if planning_lease is None:
            nav_events, nav_proposals = [], []
            next_cells = self.navigation.next_cells(self.agent)
        elif planning_lease.owner is UnitCode.PLANNING:
            nav_events, nav_proposals, next_cells = self.navigation.propose(perceived_state, self.tick)
        else:
            remembered_obstacles = {
                tuple(cell) for cell in perceived_state["obstacles"]
            } | set(self.memory.bad_cells)
            width, height = perceived_state["size"]
            candidate_cells = self.navigation.next_cells(self.agent).values()
            has_unvisited_exit = any(
                0 <= cell[0] < width and 0 <= cell[1] < height
                and cell not in remembered_obstacles and cell not in self.visited_cells
                for cell in candidate_cells
            )
            if has_unvisited_exit:
                remembered_obstacles |= self.visited_cells - {self.agent, self.goal}
            backup_state = {
                **perceived_state,
                "obstacles": [list(cell) for cell in sorted(remembered_obstacles)],
            }
            nav_events, nav_proposals, next_cells = self.backup_planners[planning_lease.owner].propose(
                backup_state, self.tick, self.adapter,
            )
        if UnitCode.PERCEPTION in self.disabled_units:
            visual_events, visual_proposals = [], []
            snn_state = {"danger": 0.0, "spike": False, "potential": self.vision.network.neuron.potential}
        else:
            visual_events, visual_proposals, snn_state = self.vision.inspect(sensor_state, self.tick)
        if UnitCode.MEMORY in self.disabled_units:
            memory_events, memory_proposals = [], []
        else:
            memory_events, memory_proposals = self.memory.recall(next_cells, self.tick)
        all_events = prediction_events + nav_events + visual_events + memory_events
        for event in all_events:
            self.bus.observe(event)

        salient = self.bus.drain()
        proposals = nav_proposals + visual_proposals + memory_proposals
        decision = self.workspace.integrate(salient, proposals)
        self.last_decision = decision

        collision = False
        if decision.action in MOVES:
            dx, dy = MOVES[decision.action]
            target = (self.agent[0] + dx, self.agent[1] + dy)
            width, height = state["size"]
            if target in self.obstacles or not (0 <= target[0] < width and 0 <= target[1] < height):
                collision = True
                self.collisions += 1
                self.memory.remember_collision(target)
                learned = self.adapter.encode_semantic(
                    UnitCode.EXECUTION, ConceptCode.COLLISION,
                    (target[0] / max(1, width - 1), target[1] / max(1, height - 1), 1.0, 1.0),
                )
                self.bus.observe(SemanticPacket(
                    source=UnitCode.EXECUTION,
                    target=UnitCode.WORKSPACE,
                    concept=ConceptCode.COLLISION,
                    tick=self.tick,
                    confidence=1.0,
                    salience=1.0,
                    urgency=1.0,
                    risk=1.0,
                    ttl_ms=500,
                    state_delta=(float(target[0]), float(target[1])),
                    latent=learned.quantized_latent,
                    semantic_code=learned.semantic_code,
                    quantization_error=learned.quantization_error,
                ))
            else:
                self.agent = target
                self.visited_cells.add(self.agent)

        if self.agent == self.goal:
            self.done = True
            learned = self.adapter.encode_semantic(
                UnitCode.EXECUTION, ConceptCode.GOAL_REACHED,
                (self.goal[0] / max(1, width - 1), self.goal[1] / max(1, height - 1), 1.0),
            )
            self.bus.observe(SemanticPacket(
                source=UnitCode.EXECUTION,
                target=UnitCode.WORKSPACE,
                concept=ConceptCode.GOAL_REACHED,
                tick=self.tick,
                confidence=1.0,
                salience=1.0,
                expected_reward=1.0,
                ttl_ms=1_000,
                state_delta=(float(self.goal[0]), float(self.goal[1])),
                latent=learned.quantized_latent,
                semantic_code=learned.semantic_code,
                quantization_error=learned.quantization_error,
            ))

        snapshot = self.snapshot()
        snapshot["transition"] = {
            "collision": collision,
            "snn": snn_state,
            "salient_events": [packet.to_trace() for packet in salient],
            "planning_owner": planning_lease.owner.name.lower() if planning_lease else None,
            "planning_delegated": planning_lease.delegated if planning_lease else False,
            "world_model": {
                "coverage": round(self.prediction.coverage, 3),
                "observed_cells": self.prediction.observed_cells,
                "known_obstacles": len(self.prediction.known_obstacles),
                "volatility": round(self.prediction.volatility, 3),
                "expired_cells": self.prediction.expired_cells,
            },
        }
        return snapshot

    def snapshot(self) -> dict:
        local_energy = (
            self.vision.energy + self.vision.network.energy
            + self.memory.energy + self.navigation.energy
            + self.prediction.energy
            + sum(planner.energy for planner in self.backup_planners.values())
        )
        return {
            **self._state(),
            "tick": self.tick,
            "done": self.done,
            "collisions": self.collisions,
            "decision": self.last_decision.to_dict() if self.last_decision else None,
            "attention": [packet.to_trace() for packet in self.workspace.attention],
            "metrics": {
                "observations": self.bus.observations,
                "published_events": self.bus.published,
                "event_sparsity": round(self.bus.sparsity, 3),
                "snn_spikes": self.vision.network.neuron.spikes,
                "local_energy": round(local_energy, 3),
                "global_energy": round(self.workspace.global_energy, 3),
                "adapter_loss": round(self.adapter.last_loss, 6),
                "adapter_mean_loss": round(self.adapter.mean_loss, 6),
                "adapter_steps": self.adapter.steps,
                "semantic_codes_used": self.adapter.quantizer.codes_used,
                "quantization_error": round(self.adapter.quantizer.last_error, 6),
                "sensor_noise": self.sensor_noise,
                "disabled_units": len(self.disabled_units),
                "capability_handoffs": self.capabilities.handoffs,
                "planning_owner": self.capabilities.assignments.get(
                    CapabilityCode.PLAN_ROUTE, UnitCode.PLANNING
                ).name.lower(),
                "backup_accuracy": round(max(
                    planner.training_accuracy for planner in self.backup_planners.values()
                ), 3),
                "partial_observability": self.partial_observability,
                "world_model_coverage": round(self.prediction.coverage, 3),
                "world_model_cells": self.prediction.observed_cells,
                "world_model_volatility": round(self.prediction.volatility, 3),
                "world_model_expired": self.prediction.expired_cells,
            },
        }

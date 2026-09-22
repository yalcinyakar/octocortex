from __future__ import annotations

import random

from octocortex.arms.memory import MemoryArm
from octocortex.arms.navigation import MOVES, NavigationArm
from octocortex.arms.vision import VisionArm
from octocortex.core.event_bus import SparseEventBus
from octocortex.core.models import ActionCode
from octocortex.core.octoir import ConceptCode, SemanticPacket, UnitCode
from octocortex.core.workspace import GlobalWorkspace
from octocortex.learning.adapter import SparseSemanticAdapter


class OctoSimulation:
    def __init__(
        self,
        seed: int = 7,
        sensor_noise: float = 0.0,
        disabled_units: frozenset[UnitCode] = frozenset(),
    ) -> None:
        if not 0.0 <= sensor_noise <= 1.0:
            raise ValueError("sensor_noise must be between 0 and 1")
        self.seed = seed
        self.sensor_noise = sensor_noise
        self.disabled_units = frozenset(disabled_units)
        self.random = random.Random(seed)
        self.reset()

    def reset(self) -> dict:
        # Reset the cognitive state as well as the body/world state so repeated
        # experiments remain comparable from the browser.
        self.random.seed(self.seed)
        self.bus = SparseEventBus(threshold=0.5)
        self.workspace = GlobalWorkspace()
        self.adapter = SparseSemanticAdapter()
        self.vision = VisionArm(self.adapter)
        self.memory = MemoryArm(self.adapter)
        self.navigation = NavigationArm(self.adapter)
        self.tick = 0
        self.agent = (0, 0)
        self.goal = (11, 7)
        self.obstacles = {
            (2, 0), (2, 1), (2, 2), (4, 2), (5, 2), (6, 2),
            (6, 3), (6, 4), (8, 4), (9, 4), (10, 4), (8, 6),
        }
        self.collisions = 0
        self.done = False
        self.last_decision = None
        return self.snapshot()

    def _state(self) -> dict:
        return {
            "size": [12, 8],
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

    def step(self) -> dict:
        if self.done:
            return self.snapshot()
        self.tick += 1
        state = self._state()
        perceived_state = self._perceived_state()

        if UnitCode.PLANNING in self.disabled_units:
            nav_events, nav_proposals = [], []
            next_cells = self.navigation.next_cells(self.agent)
        else:
            nav_events, nav_proposals, next_cells = self.navigation.propose(perceived_state, self.tick)
        if UnitCode.PERCEPTION in self.disabled_units:
            visual_events, visual_proposals = [], []
            snn_state = {"danger": 0.0, "spike": False, "potential": self.vision.network.neuron.potential}
        else:
            visual_events, visual_proposals, snn_state = self.vision.inspect(perceived_state, self.tick)
        if UnitCode.MEMORY in self.disabled_units:
            memory_events, memory_proposals = [], []
        else:
            memory_events, memory_proposals = self.memory.recall(next_cells, self.tick)
        all_events = nav_events + visual_events + memory_events
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
                    (target[0] / 11, target[1] / 7, 1.0, 1.0),
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

        if self.agent == self.goal:
            self.done = True
            learned = self.adapter.encode_semantic(
                UnitCode.EXECUTION, ConceptCode.GOAL_REACHED,
                (self.goal[0] / 11, self.goal[1] / 7, 1.0),
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
        }
        return snapshot

    def snapshot(self) -> dict:
        local_energy = (
            self.vision.energy + self.vision.network.energy
            + self.memory.energy + self.navigation.energy
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
            },
        }

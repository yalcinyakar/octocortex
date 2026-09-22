from __future__ import annotations

import random

from octocortex.arms.memory import MemoryArm
from octocortex.arms.navigation import MOVES, NavigationArm
from octocortex.arms.vision import VisionArm
from octocortex.core.event_bus import SparseEventBus
from octocortex.core.models import Event
from octocortex.core.workspace import GlobalWorkspace


class OctoSimulation:
    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.random = random.Random(seed)
        self.reset()

    def reset(self) -> dict:
        # Reset the cognitive state as well as the body/world state so repeated
        # experiments remain comparable from the browser.
        self.bus = SparseEventBus(threshold=0.5)
        self.workspace = GlobalWorkspace()
        self.vision = VisionArm()
        self.memory = MemoryArm()
        self.navigation = NavigationArm()
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

    def step(self) -> dict:
        if self.done:
            return self.snapshot()
        self.tick += 1
        state = self._state()

        nav_events, nav_proposals, next_cells = self.navigation.propose(state, self.tick)
        visual_events, visual_proposals, snn_state = self.vision.inspect(state, self.tick)
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
                self.bus.observe(Event("body", "collision", 1.0, {"cell": list(target)}, self.tick))
            else:
                self.agent = target

        if self.agent == self.goal:
            self.done = True
            self.bus.observe(Event("body", "goal_reached", 1.0, {}, self.tick))

        snapshot = self.snapshot()
        snapshot["transition"] = {
            "collision": collision,
            "snn": snn_state,
            "salient_events": [event.to_dict() for event in salient],
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
            "attention": [event.to_dict() for event in self.workspace.attention],
            "metrics": {
                "observations": self.bus.observations,
                "published_events": self.bus.published,
                "event_sparsity": round(self.bus.sparsity, 3),
                "snn_spikes": self.vision.network.neuron.spikes,
                "local_energy": round(local_energy, 3),
                "global_energy": round(self.workspace.global_energy, 3),
            },
        }

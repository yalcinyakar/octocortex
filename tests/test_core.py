import unittest

from octocortex.core.arbitrator import Arbitrator
from octocortex.core.event_bus import SparseEventBus
from octocortex.core.models import Event, Proposal
from octocortex.simulation.environment import OctoSimulation
from octocortex.snn.network import LIFNeuron


class CoreTests(unittest.TestCase):
    def test_sparse_bus_filters_low_salience(self):
        bus = SparseEventBus(threshold=0.5)
        self.assertFalse(bus.observe(Event("test", "quiet", 0.2)))
        self.assertTrue(bus.observe(Event("test", "alarm", 0.8)))
        self.assertEqual([event.kind for event in bus.drain()], ["alarm"])

    def test_safety_veto_beats_reward(self):
        decision = Arbitrator().choose([
            Proposal("navigation", "RIGHT", 0.99, expected_reward=1.0),
            Proposal("vision", "WAIT", 0.8, risk=0.95, urgency=0.9),
        ])
        self.assertEqual(decision.action, "WAIT")
        self.assertEqual(decision.winner, "vision")

    def test_lif_neuron_accumulates_and_spikes(self):
        neuron = LIFNeuron(threshold=1.0, leak=1.0)
        self.assertFalse(neuron.step(0.4))
        self.assertTrue(neuron.step(0.7))
        self.assertEqual(neuron.spikes, 1)

    def test_simulation_reaches_goal(self):
        simulation = OctoSimulation()
        for _ in range(80):
            state = simulation.step()
            if state["done"]:
                break
        self.assertTrue(state["done"])
        self.assertEqual(state["agent"], state["goal"])
        self.assertGreater(state["metrics"]["observations"], state["metrics"]["published_events"])

    def test_reset_clears_cognitive_metrics(self):
        simulation = OctoSimulation()
        simulation.step()
        state = simulation.reset()
        self.assertEqual(state["tick"], 0)
        self.assertEqual(state["metrics"]["observations"], 0)
        self.assertEqual(state["metrics"]["local_energy"], 0.0)


if __name__ == "__main__":
    unittest.main()

import unittest

from octocortex.core.arbitrator import Arbitrator
from octocortex.core.event_bus import SparseEventBus
from octocortex.core.models import ActionCode, Proposal, ReasonCode
from octocortex.core.octoir import ConceptCode, SemanticPacket, UnitCode
from octocortex.simulation.environment import OctoSimulation
from octocortex.snn.network import LIFNeuron


class CoreTests(unittest.TestCase):
    def test_sparse_bus_filters_low_salience(self):
        bus = SparseEventBus(threshold=0.5)
        quiet = SemanticPacket(UnitCode.PERCEPTION, UnitCode.WORKSPACE, ConceptCode.VISUAL_SCAN, 1, salience=0.2)
        alarm = SemanticPacket(UnitCode.SAFETY, UnitCode.WORKSPACE, ConceptCode.DANGER_SPIKE, 1, salience=0.8)
        self.assertFalse(bus.observe(quiet))
        self.assertTrue(bus.observe(alarm))
        self.assertEqual([packet.concept for packet in bus.drain()], [ConceptCode.DANGER_SPIKE])

    def test_safety_veto_beats_reward(self):
        decision = Arbitrator().choose([
            Proposal(UnitCode.PLANNING, ActionCode.RIGHT, 0.99, expected_reward=1.0),
            Proposal(UnitCode.PERCEPTION, ActionCode.WAIT, 0.8, risk=0.95, urgency=0.9, reason_code=ReasonCode.DANGER_SPIKE),
        ])
        self.assertEqual(decision.action, ActionCode.WAIT)
        self.assertEqual(decision.winner, UnitCode.PERCEPTION)

    def test_octoir_binary_round_trip(self):
        packet = SemanticPacket(
            source=UnitCode.PREDICTION,
            target=UnitCode.WORKSPACE,
            concept=ConceptCode.ROUTE_PROPOSAL,
            tick=42,
            confidence=0.91,
            uncertainty=0.09,
            salience=0.72,
            expected_reward=0.8,
            ttl_ms=250,
            state_delta=(2.0, 5.0),
            latent=(0.1, 0.0, -0.4),
        )
        decoded = SemanticPacket.decode(packet.encode())
        self.assertEqual(decoded.source, packet.source)
        self.assertEqual(decoded.concept, packet.concept)
        self.assertEqual(decoded.tick, 42)
        self.assertEqual(decoded.state_delta, packet.state_delta)
        self.assertAlmostEqual(decoded.latent[2], -0.4, places=6)

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

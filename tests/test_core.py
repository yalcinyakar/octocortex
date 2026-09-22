import unittest

from benchmarks.architecture_comparison import run_benchmark
from octocortex.core.arbitrator import Arbitrator
from octocortex.core.capabilities import CapabilityCode, CapabilityRouter
from octocortex.core.event_bus import SparseEventBus
from octocortex.core.models import ActionCode, Proposal, ReasonCode
from octocortex.core.octoir import ConceptCode, SemanticPacket, UnitCode
from octocortex.simulation.environment import OctoSimulation
from octocortex.learning.adapter import SparseSemanticAdapter
from octocortex.learning.quantizer import OnlineVectorQuantizer
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
            semantic_code=5,
            quantization_error=0.031,
        )
        decoded = SemanticPacket.decode(packet.encode())
        self.assertEqual(decoded.source, packet.source)
        self.assertEqual(decoded.concept, packet.concept)
        self.assertEqual(decoded.tick, 42)
        self.assertEqual(decoded.state_delta, packet.state_delta)
        self.assertAlmostEqual(decoded.latent[2], -0.4, places=6)
        self.assertEqual(decoded.semantic_code, 5)
        self.assertAlmostEqual(decoded.quantization_error, 0.031, places=6)

    def test_lif_neuron_accumulates_and_spikes(self):
        neuron = LIFNeuron(threshold=1.0, leak=1.0)
        self.assertFalse(neuron.step(0.4))
        self.assertTrue(neuron.step(0.7))
        self.assertEqual(neuron.spikes, 1)

    def test_sparse_adapter_learns_and_limits_active_latents(self):
        adapter = SparseSemanticAdapter()
        vector = adapter.semantic_vector(UnitCode.PERCEPTION, ConceptCode.DANGER_SPIKE, (0.8, 1.0, 0.5))
        losses = [adapter.encode_and_learn(vector).loss for _ in range(200)]
        result = adapter.encode_and_learn(vector)
        self.assertLess(sum(losses[-20:]) / 20, sum(losses[:20]) / 20)
        self.assertLessEqual(sum(value != 0 for value in result.latent), 2)
        self.assertEqual(adapter.checkpoint()["steps"], 201)

    def test_vector_quantizer_assigns_stable_codes(self):
        quantizer = OnlineVectorQuantizer()
        vector = (0.0, -0.2, -0.18, 0.0)
        codes = [quantizer.quantize(vector).code for _ in range(100)]
        self.assertEqual(len(set(codes[-20:])), 1)
        self.assertLess(quantizer.quantize(vector, learn=False).error, 0.001)
        self.assertGreaterEqual(quantizer.codes_used, 1)

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
        self.assertEqual(state["metrics"]["adapter_steps"], 0)
        self.assertEqual(state["metrics"]["semantic_codes_used"], 0)

    def test_noise_replay_is_deterministic(self):
        first = OctoSimulation(seed=31, sensor_noise=0.2)
        second = OctoSimulation(seed=31, sensor_noise=0.2)
        first_path = [first.step()["agent"] for _ in range(12)]
        second_path = [second.step()["agent"] for _ in range(12)]
        self.assertEqual(first_path, second_path)

    def test_planning_capability_fails_over_to_memory(self):
        router = CapabilityRouter()
        lease = router.resolve(
            CapabilityCode.PLAN_ROUTE,
            {UnitCode.MEMORY, UnitCode.PERCEPTION},
        )
        self.assertEqual(lease.owner, UnitCode.MEMORY)
        self.assertTrue(lease.delegated)
        self.assertEqual(router.handoffs, 1)

    def test_simulation_reaches_goal_after_planning_failure(self):
        simulation = OctoSimulation(disabled_units=frozenset({UnitCode.PLANNING}))
        for _ in range(80):
            state = simulation.step()
            if state["done"]:
                break
        self.assertTrue(state["done"])
        self.assertEqual(state["metrics"]["planning_owner"], "memory")
        self.assertEqual(state["metrics"]["capability_handoffs"], 1)

    def test_architecture_benchmark_covers_all_cases(self):
        report = run_benchmark(trials=3, max_steps=80)
        self.assertEqual(len(report["summary"]), 12)
        clean = [row for row in report["summary"] if row["scenario"] == "clean"]
        self.assertTrue(all(row["success_rate"] == 1.0 for row in clean))


if __name__ == "__main__":
    unittest.main()

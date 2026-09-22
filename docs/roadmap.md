# OctoCortex technical roadmap

The goal is a sparse, fault-tolerant cognitive runtime whose specialized units
exchange compact OctoIR messages. Progress is gated by falsifiable benchmarks,
not feature count. The protocol topology remains fixed; new behavior should
fill or improve existing roles rather than add ad-hoc arms.

## Stage 1 — Executable substrate (complete)

- Sparse event bus, global workspace, arbitration, SNN danger signal.
- Binary OctoIR packets, learned sparse adapter, eight-code quantizer.
- Browser observability for local work, proposals, events, and energy proxy.

Exit evidence: deterministic tests, OctoIR round trip, browser/Python parity.

## Stage 2 — Fault-tolerant specialization (complete)

- Dynamic planning-capability lease.
- Independently parameterized Memory and Perception backup policies.
- Primary-planner ablation proving failover does not call the primary code.

Exit evidence: 100% success under the current component-failure scenario.

## Stage 3 — Held-out transfer (complete)

- Procedural world generator.
- 32-world training / 100-world test split with fingerprint overlap checks.
- Fixed demo world excluded from distillation.

Exit evidence: 98% held-out success for both backup owners and 99% action
agreement. Scope is monotonic-route worlds only.

## Stage 4A — Static world model under partial observability (complete)

- Replace full-map input with a local sensor window.
- Add a recurrent Prediction unit that estimates occupancy and uncertainty.
- Require detours and temporary movement away from the goal.

Exit evidence: 100% success on 100 unseen static detour worlds versus 0% for
the local-only memory ablation, with no full-map access.

## Stage 4B — Dynamic uncertainty (next)

- Add moving obstacles, observation drift, and disappearing paths.
- Learn transition dynamics instead of storing occupancy alone.
- Calibrate confidence and add out-of-distribution detection to OctoIR.

Exit gate: at least 90% success on unseen dynamic worlds, bounded collision
rate, calibrated uncertainty, and a statistically reported ablation gap.

## Stage 5 — Equal-budget baselines

- Compare against A*, tabular Q-learning/PPO, a small Transformer policy, and a
  mixture-of-experts controller under matched observations and step budgets.
- Report parameters, training samples, latency, memory, logical energy, and
  confidence intervals. Do not claim hardware efficiency from Python runtime.

Exit gate: OctoCortex must win on a declared Pareto axis—fault tolerance,
sample efficiency, energy, or transfer—without losing unacceptable accuracy.

## Stage 6 — Asynchronous cognitive runtime

- Version OctoIR schemas and add replay/idempotency semantics.
- Put hot working state in an in-memory store; add a Kafka-compatible durable
  event log only after single-process semantics are stable.
- Run units at independent rates with backpressure, deadlines, and lease
  recovery instead of one synchronous tick loop.

Exit gate: deterministic replay, no duplicated decisions, bounded queue growth,
and recovery after process loss.

## Stage 7 — Real task integration

- Connect one non-grid environment such as a robotics simulator or tool-using
  software agent.
- Keep OctoIR internal; natural language is an edge adapter, not the cognitive
  transport.
- Measure end-to-end usefulness against a monolithic model with the same tools.

Exit gate: reproducible external task gains and an independently runnable
benchmark package.

## Research discipline

- Every new capability needs an ablation.
- Training and evaluation data must remain disjoint and fingerprinted.
- Dashboard numbers must be generated from checked-in benchmark code.
- "Better than existing models" requires matched baselines and confidence
  intervals; architecture diagrams or toy success rates are not sufficient.

# Architecture benchmark methodology

The benchmark compares three control organizations on the same 12×8 grid,
obstacles, goal, maximum 80-step budget, seeds, sensor-noise rate, and injected
component outage.

- **Centralized:** one greedy controller; an injected controller outage stops
  the whole policy.
- **Distributed:** planning, reactive perception, and collision-memory voters
  choose an action without a global workspace.
- **OctoCortex:** the production simulation uses its sparse event bus, learned
  adapter, vector quantizer, proposals, global workspace, and capability
  router. Route planning is leased to memory or perception if its primary unit
  is unavailable.

## Scenarios

| Scenario | Sensor noise | Component outage probability |
| --- | ---: | ---: |
| clean | 0% | 0% |
| sensor noise | 15% | 0% |
| component failure | 0% | 35% |
| combined | 15% | 35% |

Noise flips perceived occupancy for adjacent cells. The real world remains
unchanged, so a false-negative perception can produce a real collision. A
component outage lasts for the episode. All architectures receive the same
deterministic failure selection for a given seed.

## Metrics and limits

Success rate, successful-episode steps, collisions, logical energy, and Python
runtime are reported. Logical energy is a declared operation-cost proxy rather
than a physical measurement. Python runtime compares these reference
implementations only; it does not establish hardware efficiency. The embedded
dashboard table is a checked-in result from 100 seeds and must be regenerated
when policies, costs, or scenarios change.

The current failover transfers ownership to independently parameterized backup
policies. Memory and perception each use four action-specific linear heads with
five features per head. Ablation tests replace the primary planner with a
deliberate failure and verify that the backup still reaches the goal.

## Held-out world transfer

`benchmarks.generalization` separates 32 procedural training worlds (seeds
1000–1031) from 100 test worlds (seeds 10000–10099). World fingerprints are
checked for zero overlap. The fixed dashboard map is not part of training.

| Split | Owner | Worlds | Success | Steps | Collisions | Action agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | memory | 32 | 96.9% | 18.39 | 0.031 | 100% |
| train | perception | 32 | 93.8% | 18.33 | 0.062 | 100% |
| unseen | memory | 100 | 98.0% | 18.10 | 0.020 | 99% |
| unseen | perception | 100 | 98.0% | 18.20 | 0.020 | 99% |

The generator guarantees that each free cell has at least one neighbor closer
to the goal. These results therefore demonstrate out-of-sample transfer of the
local policy, not maze solving, long-horizon credit assignment, or general
intelligence. Safety spikes remain observable but issue a hard WAIT veto only
when all four neighboring cells are occupied; proximity alone cannot deadlock
the agent.

## Partial observability and recurrent world model

`benchmarks.partial_observability` removes full-map access. At each step the
agent receives only a 3×3 window centered on its current position. The
Prediction unit accumulates those local observations into a recurrent occupancy
belief, reports coverage and uncertainty in OctoIR, and gives the route planner
only the current belief map.

The 100 held-out hard worlds contain a wall whose gap is on the side opposite
the goal, so the route must temporarily increase goal distance. With the same
planner but without accumulated prediction state, the agent forgets the wall
between observations and oscillates.

| Condition | Worlds | Success | Steps | Collisions | Mean map coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| local observation only | 100 | 0% | — | 0.000 | 0% |
| OctoCortex world model | 100 | 100% | 27.40 | 0.000 | 63.6% |

This isolates the value of persistent spatial belief on a controlled static
task. It does not yet cover moving obstacles, observation drift, learned
dynamics, or calibrated out-of-distribution detection.

## Dynamic uncertainty

`benchmarks.dynamic_uncertainty` varies wall position, route orientation, and
change timing across 100 seeded trials. It removes a gate after observation
while it is outside the 3×3 sensor window. Local observations also have a 2%
occupancy-flip rate. The static-belief ablation retains every past obstacle
indefinitely. The adaptive model learns an exponential change rate, shortens
its stale horizon in volatile scenes, expires obsolete cells, and lowers
confidence as volatility rises. OctoIR flag `4` marks elevated world-model
volatility.

| Belief policy | Trials | Success | Steps | Collisions | Expired cells | Peak volatility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| persistent static belief | 100 | 58% | 18.83 | 0.260 | 0.00 | 0.079 |
| adaptive dynamic belief | 100 | 96% | 17.96 | 0.040 | 27.72 | 0.062 |

This is still a controlled disappearing-gate task, not a general learned
physics model. It establishes that age- and volatility-aware belief revision is
materially better than permanent memory under change and sensor drift.

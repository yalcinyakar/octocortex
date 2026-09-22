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

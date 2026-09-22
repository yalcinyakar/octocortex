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
policies. Memory and perception each distill route examples into their own
five-weight linear model before execution. Ablation tests replace the primary
planner with a deliberate failure and verify that the backup still reaches the
goal. This demonstrates independent execution, but not yet generalization to
maps absent from the distillation set.

# OctoCortex

OctoCortex is a small, executable experiment in hybrid AI architecture:

- a **global workspace** integrates only salient events;
- semi-autonomous **arms** perceive, remember, and propose actions;
- a tiny **spiking neural network (SNN)** turns local sensory pressure into sparse events;
- an **arbitrator** chooses between proposals using confidence, reward, risk, urgency, and energy cost.

Processing units communicate through **OctoIR**, a typed language-free
intermediate representation. Human-readable labels are decoded only at the
dashboard boundary; the internal event fabric transports numeric concept IDs,
state deltas, confidence/risk signals, and sparse latent vectors.
One shared online autoencoder now learns those latent vectors directly from
typed local signals. Its top-k bottleneck keeps only two of four latent
components active, avoiding text/token conversion between processing units.
An online vector quantizer maps each sparse latent to one of eight shared
semantic codes, so similar internal states can converge on the same compact
identity across units.

The first demo is a grid-world agent. The navigation arm tries to reach a goal, the visual arm emits danger spikes near obstacles, and the memory arm learns which cells have caused collisions. The browser UI exposes every proposal and arbitration decision.

## Live demo

Run OctoCortex directly in the browser: <https://yalcinyakar.github.io/octocortex/>

## Run

Requires Python 3.11+ and no third-party packages.

```bash
python -m octocortex.web
```

Then open <http://127.0.0.1:8000>.

## Test

```bash
python -m unittest discover -s tests -v
```

Compare the compact OctoIR wire packet with its human-readable trace:

```bash
python -m benchmarks.octoir_size
```

Compare centralized, fully distributed, and OctoCortex control over 100
deterministic seeds and four fault scenarios:

```bash
python -m benchmarks.architecture_comparison --trials 100
```

The energy number is a logical operation-cost proxy, not joules. Runtime is
implementation-specific Python wall time and must not be presented as a
hardware performance result.

## Project shape

```text
octocortex/
├── arms/          # local perception, memory, and navigation policies
├── core/          # event bus, proposals, workspace, arbitration
├── docs/          # OctoIR protocol specification
├── learning/      # shared online sparse semantic adapter
├── simulation/    # deterministic grid-world experiment
├── snn/           # minimal leaky-integrate-and-fire network
├── static/        # live dashboard
└── web.py         # dependency-free HTTP API
```

## Current experiment

The architecture is intentionally measurable. The API reports:

- goal progress and collisions;
- published event count versus raw local observations;
- chosen arm and decision score;
- cumulative local and global energy estimates;
- SNN spikes and membrane potential.
- semantic adapter reconstruction loss and training steps.
- vector-quantization error and shared semantic-code utilization.

This is an architecture simulator, not yet a biological fidelity claim. The
current benchmark shows lower logical energy for OctoCortex, but also exposes a
planning-arm dependency: success falls under injected component failure. The
next milestone is redundant planning and dynamic arm reassignment.

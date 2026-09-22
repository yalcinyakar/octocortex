# OctoCortex

OctoCortex is a small, executable experiment in hybrid AI architecture:

- a **global workspace** integrates only salient events;
- semi-autonomous **arms** perceive, remember, and propose actions;
- a tiny **spiking neural network (SNN)** turns local sensory pressure into sparse events;
- an **arbitrator** chooses between proposals using confidence, reward, risk, urgency, and energy cost.

The first demo is a grid-world agent. The navigation arm tries to reach a goal, the visual arm emits danger spikes near obstacles, and the memory arm learns which cells have caused collisions. The browser UI exposes every proposal and arbitration decision.

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

## Project shape

```text
octocortex/
├── arms/          # local perception, memory, and navigation policies
├── core/          # event bus, proposals, workspace, arbitration
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

This is an architecture simulator, not yet a biological fidelity claim. The next useful milestone is a benchmark harness comparing this hybrid against a centralized policy and a fully distributed policy under sensor noise and arm failure.


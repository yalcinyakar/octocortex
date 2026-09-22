# OctoIR v0.1

OctoIR is the language-free intermediate representation used between
OctoCortex processing units. Natural language exists only at system boundaries
for human input, output, and observability.

## Packet

Each packet has a fixed typed header followed by two float vectors:

| Field | Purpose |
| --- | --- |
| `source`, `target` | Numeric processing-unit identifiers |
| `concept` | Numeric semantic concept identifier |
| `tick`, `ttl_ms` | Logical time and expiry |
| `confidence`, `uncertainty` | Epistemic quality |
| `salience`, `urgency`, `risk` | Routing and arbitration signals |
| `expected_reward` | Local utility estimate |
| `state_delta` | Explicit low-dimensional state change |
| `latent` | Sparse learned representation |

The binary wire format starts with `OIR1`, is little-endian, and contains no
JSON keys, prose, tokenizer output, or human-readable module names.

## Boundary rule

Core units may encode, route, decode, and learn from OctoIR packets. They must
not communicate using `to_trace()` output. That decoder exists solely for the
dashboard, logs, and debugging tools.

## Learned semantic adapter

All processing units share one online autoencoder. Each unit supplies a typed
eight-value vector: numeric concept, numeric source, and up to six normalized
local features. The encoder produces four latent values and retains only the
two largest-magnitude activations. The decoder reconstructs the input and both
matrices learn immediately with stochastic gradient descent.

This reference adapter is dependency-free and deliberately small. It proves
the architectural contract: units exchange a learned sparse representation,
while OctoIR remains stable if the adapter is later replaced by a larger
PyTorch or JAX model. The dashboard exposes last and mean reconstruction loss;
human-readable labels remain boundary-only.

## Evolution

Version 0.1 establishes transport, observability, and learned adapter
boundaries. Later versions will add vector quantization, schema negotiation, causal
references, and compatibility rules for a Kafka-compatible event fabric.

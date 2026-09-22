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

## Evolution

Version 0.1 establishes transport and observability boundaries. Later versions
will add learned adapters, vector quantization, schema negotiation, causal
references, and compatibility rules for a Kafka-compatible event fabric.

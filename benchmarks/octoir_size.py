"""Compare an OctoIR packet with its human-readable trace representation."""

import json

from octocortex.core.octoir import ConceptCode, SemanticPacket, UnitCode


packet = SemanticPacket(
    source=UnitCode.PERCEPTION,
    target=UnitCode.WORKSPACE,
    concept=ConceptCode.DANGER_SPIKE,
    tick=42,
    confidence=0.97,
    uncertainty=0.03,
    salience=0.94,
    urgency=0.90,
    risk=0.93,
    ttl_ms=180,
    state_delta=(2.0,),
    latent=(0.12, 0.80),
)

wire = packet.encode()
trace = json.dumps(packet.to_trace(), separators=(",", ":")).encode()
reduction = 1.0 - len(wire) / len(trace)

print(f"OctoIR binary: {len(wire)} bytes")
print(f"JSON trace:    {len(trace)} bytes")
print(f"Reduction:     {reduction:.1%}")

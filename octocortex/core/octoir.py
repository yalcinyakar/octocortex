from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum


class UnitCode(IntEnum):
    SYSTEM = 0
    PERCEPTION = 1
    MEMORY = 2
    PREDICTION = 3
    PLANNING = 4
    SAFETY = 5
    EXECUTION = 6
    WORKSPACE = 7


class ConceptCode(IntEnum):
    VISUAL_SCAN = 1
    DANGER_SPIKE = 2
    ROUTE_PROPOSAL = 3
    TRAPPED = 4
    MEMORY_QUIET = 5
    COLLISION_RECALL = 6
    COLLISION = 7
    GOAL_REACHED = 8
    WORLD_MODEL_UPDATE = 9


_HEADER_V1 = struct.Struct("<4sBBBHBIIffffffHH")
_HEADER_V2 = struct.Struct("<4sBBBHHBIIfffffffHH")
_MAGIC = b"OIR1"


@dataclass(frozen=True, slots=True)
class SemanticPacket:
    """Language-free message exchanged by OctoCortex processing units.

    `state_delta` contains explicit low-dimensional state changes. `latent`
    carries a sparse learned representation. Human-readable labels are added
    only by `to_trace`, at the observability boundary.
    """

    source: UnitCode
    target: UnitCode
    concept: ConceptCode
    tick: int
    semantic_code: int = 0
    quantization_error: float = 0.0
    confidence: float = 1.0
    uncertainty: float = 0.0
    salience: float = 0.0
    urgency: float = 0.0
    risk: float = 0.0
    expected_reward: float = 0.0
    ttl_ms: int = 0
    state_delta: tuple[float, ...] = ()
    latent: tuple[float, ...] = ()
    flags: int = 0

    def encode(self) -> bytes:
        state = tuple(float(value) for value in self.state_delta)
        latent = tuple(float(value) for value in self.latent)
        header = _HEADER_V2.pack(
            _MAGIC,
            2,
            int(self.source),
            int(self.target),
            int(self.concept),
            self.semantic_code,
            self.flags,
            self.tick,
            self.ttl_ms,
            self.confidence,
            self.uncertainty,
            self.salience,
            self.urgency,
            self.risk,
            self.expected_reward,
            self.quantization_error,
            len(state),
            len(latent),
        )
        values = state + latent
        return header + (struct.pack(f"<{len(values)}f", *values) if values else b"")

    @classmethod
    def decode(cls, data: bytes) -> "SemanticPacket":
        if len(data) < _HEADER_V1.size:
            raise ValueError("OctoIR packet is shorter than its header")
        magic, version = struct.unpack_from("<4sB", data)
        if magic != _MAGIC or version not in (1, 2):
            raise ValueError("Unsupported OctoIR packet")
        if version == 1:
            (
                _, _, source, target, concept, flags, tick, ttl_ms,
                confidence, uncertainty, salience, urgency, risk, reward,
                state_len, latent_len,
            ) = _HEADER_V1.unpack_from(data)
            semantic_code, quantization_error, header_size = 0, 0.0, _HEADER_V1.size
        else:
            (
                _, _, source, target, concept, semantic_code, flags, tick, ttl_ms,
                confidence, uncertainty, salience, urgency, risk, reward,
                quantization_error, state_len, latent_len,
            ) = _HEADER_V2.unpack_from(data)
            header_size = _HEADER_V2.size
        count = state_len + latent_len
        expected_size = header_size + count * 4
        if len(data) != expected_size:
            raise ValueError("OctoIR payload length does not match its header")
        values = struct.unpack_from(f"<{count}f", data, header_size) if count else ()
        return cls(
            source=UnitCode(source), target=UnitCode(target),
            concept=ConceptCode(concept), tick=tick,
            semantic_code=semantic_code, quantization_error=quantization_error,
            confidence=confidence, uncertainty=uncertainty,
            salience=salience, urgency=urgency, risk=risk,
            expected_reward=reward, ttl_ms=ttl_ms,
            state_delta=tuple(values[:state_len]),
            latent=tuple(values[state_len:]), flags=flags,
        )

    def to_trace(self) -> dict:
        """Decode a packet for humans; never used for unit-to-unit transport."""
        return {
            "source": self.source.name.lower(),
            "kind": self.concept.name.lower(),
            "salience": round(self.salience, 4),
            "payload": {
                "concept_id": int(self.concept),
                "semantic_code": self.semantic_code,
                "quantization_error": round(self.quantization_error, 6),
                "confidence": round(self.confidence, 4),
                "uncertainty": round(self.uncertainty, 4),
                "urgency": round(self.urgency, 4),
                "risk": round(self.risk, 4),
                "state_delta": [round(value, 4) for value in self.state_delta],
                "latent": [round(value, 4) for value in self.latent],
            },
            "tick": self.tick,
        }

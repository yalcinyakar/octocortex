"""Trainable representation adapters for OctoCortex."""

from .adapter import SparseSemanticAdapter
from .quantizer import OnlineVectorQuantizer, QuantizedResult

__all__ = ["OnlineVectorQuantizer", "QuantizedResult", "SparseSemanticAdapter"]

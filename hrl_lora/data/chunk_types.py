from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class FrameBatch:
    """Single-frame batch used for Router observation."""

    batch: Dict[str, torch.Tensor]
    meta: Dict[str, Any]


@dataclass
class ChunkBatch:
    """Chunk batch (chunk_size action horizon) used for reward + LLP supervised update."""

    batch: Dict[str, torch.Tensor]
    time_progress: torch.Tensor  # (B,)
    done: torch.Tensor  # (B,) bool
    meta: Dict[str, Any]

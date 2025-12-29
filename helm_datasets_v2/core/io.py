# helm_datasets_v2/core/io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json


def write_jsonl_sharded(
    rows: Iterable[Dict[str, Any]],
    out_dir: Path,
    split: str,
    shard_size: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    if shard_size is None or shard_size <= 0:
        p = out_dir / f"{split}-00000.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths.append(p)
        return paths

    shard_idx = 0
    count = 0
    f = None
    p = None

    def open_new():
        nonlocal f, p, shard_idx, count
        if f is not None:
            f.close()
        p = out_dir / f"{split}-{shard_idx:05d}.jsonl"
        f = p.open("w", encoding="utf-8")
        paths.append(p)
        shard_idx += 1
        count = 0

    open_new()
    for r in rows:
        if count >= shard_size:
            open_new()
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        count += 1

    if f is not None:
        f.close()
    return paths
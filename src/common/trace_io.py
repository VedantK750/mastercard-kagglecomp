"""JSONL read/write for AttackTrace — the one on-disk format every Red
generator writes and every Blue detector / eval harness reads."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from .schemas import AttackTrace


def write_traces(traces: List[AttackTrace], path: str | Path, mode: str = "w") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for trace in traces:
            f.write(trace.model_dump_json() + "\n")


def append_trace(trace: AttackTrace, path: str | Path) -> None:
    write_traces([trace], path, mode="a")


def read_traces(path: str | Path) -> Iterator[AttackTrace]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield AttackTrace.model_validate_json(line)


def load_traces(path: str | Path) -> List[AttackTrace]:
    return list(read_traces(path))

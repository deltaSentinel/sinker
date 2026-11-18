"""
Feed wiring (stubbed for stage 1 — flesh out in stage 2).

Each fetcher downloads a snapshot into data/raw/ with a dated filename.
Snapshots are immutable once written; regeneration creates NEW files,
so old training sets stay reproducible.

TODO(stage 2):
  - fetch_tranco(out_dir)   top-N benign domains -> benign snapshot
  - fetch_phishtank(...)    verify feed availability + license first
  - fetch_openphish(...)    same caveat
  - dedupe against previously seen URLs, track per-feed coverage
"""
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def ensure_raw_dir() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR

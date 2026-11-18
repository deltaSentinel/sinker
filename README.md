![CI](https://github.com/deltaSentinel/sinker/actions/workflows/ci.yml/badge.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)(./README.md)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)
[![Security](https://img.shields.io/badge/security-policy%20%E2%9C%93-green.svg)](./SECURITY.md)
![Status](https://img.shields.io/badge/stage-1%20%EF%81%83%20in%20progress-orange.svg)

Multi-signal phishing detection system. Each *signal* is an independently
trained detector (URL string, HTML/DOM, email headers, page screenshot, ...)
whose probability output is combined by a fusion/arbitration layer.

Philosophy: same layered pattern as AutoSecure — cheap detectors and learned
models under a scoring umbrella. Every signal owns its full path:
ingestion -> features -> train -> export.

## Roadmap

| Stage | Deliverable | Status |
|-------|-------------|--------|
| 0 | Repo, env, layout, seeds, tests | done |
| 1 | Signal: URL char-CNN classifier + predict CLI | **in progress** |
| 2 | Dataset versioning + drift eval (feed re-snapshots) | planned |
| 3 | Signal: HTML/DOM feature detector | planned |
| 4 | Fusion/arbitration layer over signals | planned |

## Usage (Stage 1)

    uv sync
    # place data files under data/raw/  (url<TAB>label,  label 0=benign 1=phish)
    uv run python -m sinker.train --data data/raw/urls.tsv
    uv run python -m sinker.predict https://example.com/login

## Data sources (to be wired up in sinker/feeds.py)

- Benign: Tranco top-sites list (tranco-list.eu)
- Phishing: PhishTank / OpenPhish live feeds — verify current availability
  and license terms when wiring up. Mix live feeds so the dataset regenerates
  instead of going stale.

## Conventions

- Fixed seeds everywhere (see sinker/seed.py). No notebook-only results.
- All datasets are versioned snapshots under data/; never edit processed data
  by hand. Regenerate from raw + code instead.
- Signals declare a common interface: predict_proba(url) -> float in [0, 1].
- pytest smoke tests must stay green in CI; a red test blocks a merge.

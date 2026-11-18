"""
CLI scorer — makes stage 1 a tool, not a notebook.

    uv run python -m sinker.predict https://bad.example/login
    uv run python -m sinker.predict --file urls.txt
"""
from __future__ import annotations

import argparse

import torch

from . import set_seed
from .data import encode_url
from .model import HookCNN
from .train import OUT_DIR


def load_model(ckpt_path=None):
    path = ckpt_path or (OUT_DIR / "hook.pt")
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path} — run sinker.train first")
    blob = torch.load(path, map_location="cpu")
    model = HookCNN(**blob.get("config", {}))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def score_urls(model, urls):
    xs = torch.from_numpy(
        __import__("numpy").stack([encode_url(u) for u in urls])
    )
    logits = model(xs)
    return torch.sigmoid(logits).tolist()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sinker.predict")
    ap.add_argument("urls", nargs="*", help="one or more URLs to score")
    ap.add_argument("--file", help="file of URLs, one per line")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--ckpt", default=None, help="path to hook.pt")
    args = ap.parse_args(argv)

    urls = list(args.urls)
    if args.file:
        with open(args.file) as fh:
            urls += [ln.strip() for ln in fh if ln.strip()]
    if not urls:
        ap.error("provide URLs or --file")

    set_seed()
    model = load_model(args.ckpt)
    probs = score_urls(model, urls)
    for url, p in zip(urls, probs):
        verdict = "PHISH" if p >= args.threshold else "ok"
        print(f"{p:6.4f}  [{verdict:<5}]  {url}")


if __name__ == "__main__":
    main()


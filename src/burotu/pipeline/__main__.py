"""Run one pipeline stage: python -m burotu.pipeline {status|ingest|refine|train|evaluate|publish}."""

from __future__ import annotations

import json
import runpy
import sys

from burotu.config import STORE

CPU_MODELS = {
    "classic": ["burotu.models.classic.train", "burotu.models.classic.export"],
    "logreg": ["burotu.models.logreg.train"],
}
CLOUD_MODELS = {
    "glimmer": "GitHub Actions > Train Muse (task=train) on Modal",
    "cpgpt": "GitHub Actions > Train Muse (task=cpgpt) on Modal",
}


def _run(module: str, *args: str) -> None:
    print(f"$ python -m {module} {' '.join(args)}", flush=True)
    saved = sys.argv
    sys.argv = [module, *args]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = saved


def status() -> None:
    from burotu.models.registry import models

    print(f"Store: {STORE}")
    raw = [p for p in (STORE / "raw").glob("*/*")] if (STORE / "raw").exists() else []
    print(f"1 ingest    {len(raw)} raw datasets under raw/ (workstation: GEO/, GDC/, Zenodo/, figshare/)")
    for v in sorted((STORE / "refined").glob("v*")):
        st = v / "status.json"
        if st.exists():
            s = json.loads(st.read_text())
            counts: dict[str, int] = {}
            for x in s.values():
                counts[x["state"]] = counts.get(x["state"], 0) + 1
            print(f"2 refine    {v.name}: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
        else:
            print(f"2 refine    {v.name}: present")
    for m in models():
        where = STORE / "models" / m.id / m.version
        print(f"3 train     {m.id:10} {m.version:24} {'local' if where.exists() else m.weights}")
    print("4 evaluate  reports/*.json (see the admin Map page)")
    print("5 publish   python -m burotu.store publish models")


def main(argv: list[str]) -> None:
    stage, args = (argv[0], argv[1:]) if argv else ("status", [])
    if stage == "status":
        status()
    elif stage == "ingest":
        for module in ("burotu.data.gdc", "burotu.data.geo", "burotu.data.public"):
            _run(module)
        print("Workstation: D:\\Burotu\\Data\\tools\\download_all.ps1 downloads the large sets (GDC, GEO, Loyfer).")
    elif stage == "refine":
        _run("burotu.pipeline.refine", "--store", str(STORE), *args)
    elif stage == "train":
        for name in args or list(CPU_MODELS):
            if name in CLOUD_MODELS:
                print(f"{name}: trains on GPU, run {CLOUD_MODELS[name]}")
                continue
            for module in CPU_MODELS[name]:
                _run(module)
    elif stage == "evaluate":
        _run("burotu.evaluation.public_plasma")
    elif stage == "publish":
        _run("burotu.store", "publish", "models")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])

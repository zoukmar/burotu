"""Generate simulated lab files (burotu-bmc format) for the demo.

Usage: python scripts/make_demo_samples.py --out data/demo_samples
Samples come from the test split: never seen during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from burotu.core.labformat import write_lab_file
from burotu.data.reference import Reference
from burotu.data.simulate import SimConfig, simulate_cohort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo_samples")
    ap.add_argument("--healthy", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--tf", type=float, default=0.02, help="tumour fraction of cancer cases")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ref = Reference.load()
    meth, total, meta = simulate_cohort(
        ref, "test", args.healthy, args.per_class, SimConfig(seed=args.seed), tf_values=[args.tf]
    )
    out = Path(args.out)
    for i in range(len(meta)):
        sid = f"BRT-DEMO-{i:03d}"
        write_lab_file(
            out / f"{sid}.tsv",
            sid,
            ref.panel,
            meth[i],
            total[i],
            lab="Partner lab (simulated)",
            assay="targeted-bisulfite-panel-v1",
        )
    # Ground truth is kept apart from the lab files (the lab does not know it).
    meta.assign(sample_id=[f"BRT-DEMO-{i:03d}" for i in range(len(meta))]).to_csv(out / "_truth.csv", index=False)
    print(f"{len(meta)} files written to {out}")


if __name__ == "__main__":
    main()

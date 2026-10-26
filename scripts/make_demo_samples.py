"""Génère des fichiers « labo » simulés (format burotu-bmc) pour la démo.

Usage : python scripts/make_demo_samples.py --out data/demo_samples
Les échantillons viennent du split test : jamais vus à l'entraînement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from burotu.data.reference import Reference
from burotu.labformat import write_lab_file
from burotu.simulate.cfdna import SimConfig, simulate_cohort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo_samples")
    ap.add_argument("--healthy", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--tf", type=float, default=0.02, help="fraction tumorale des cas cancer")
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
            lab="Laboratoire partenaire (simulé)",
            assay="targeted-bisulfite-panel-v1",
        )
    # La vérité terrain reste séparée des fichiers labo (le labo ne la connaît pas).
    meta.assign(sample_id=[f"BRT-DEMO-{i:03d}" for i in range(len(meta))]).to_csv(out / "_truth.csv", index=False)
    print(f"{len(meta)} fichiers écrits dans {out}")


if __name__ == "__main__":
    main()

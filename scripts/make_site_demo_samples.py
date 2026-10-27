"""Curated lab files bundled with the tracking site, for the live demo.

    python scripts/make_site_demo_samples.py

All samples come from the test split (people and tumours no model was trained on).
One sample is sequenced too shallowly on purpose, to show quality control at work.
Ground truth is kept in manifest.json and shown only after an analysis, as "simulation
ground truth", so the demo can be checked honestly.
"""

from __future__ import annotations

import json

import numpy as np

from burotu.config import HEALTHY, ROOT
from burotu.data.reference import Reference
from burotu.labformat import write_lab_file
from burotu.simulate.cfdna import SimConfig, mix_and_sequence

OUT = ROOT / "src" / "burotu_web" / "demo_samples"

# (label, tumour fraction, sequencing depth)
CASES = [
    (HEALTHY, 0.0, 500),
    ("Colorectal", 0.03, 500),
    ("Lung", 0.03, 500),
    (HEALTHY, 0.0, 500),
    ("Pancreas", 0.10, 500),
    ("Breast", 0.01, 500),
    ("Liver", 0.03, 500),
    ("Colorectal", 0.03, 40),  # under-sequenced: quality control should stop it
]


def main() -> None:
    ref = Reference.load()
    rng = np.random.default_rng(2026)
    blood = ref.blood_test.to_numpy(np.float32)
    tumors = ref.tumor_test
    meta = ref.tumor_meta.set_index("file_id").loc[tumors.columns]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (label, tf, depth) in enumerate(CASES, start=1):
        donor = int(rng.integers(blood.shape[1]))
        tumor = None
        if label != HEALTHY:
            idx = np.flatnonzero(meta["cancer_class"].to_numpy() == label)
            tumor = tumors.to_numpy(np.float32)[:, int(rng.choice(idx))]
        meth, total = mix_and_sequence(blood[:, donor], tumor, tf, SimConfig(depth=depth), rng)
        sid = f"BRT-DEMO-{i:02d}"
        write_lab_file(
            OUT / f"{sid}.tsv",
            sid,
            ref.panel,
            meth,
            total,
            lab="Partner lab (simulated)",
            assay="targeted-bisulfite-panel-v1",
        )
        manifest.append({"sample_id": sid, "label": label, "tumor_fraction": tf, "depth": depth})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} demo samples written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

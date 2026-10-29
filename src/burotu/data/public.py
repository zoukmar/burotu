"""Open public methylation datasets from GEO, cut down to Burotu's panel.

    python -m burotu.data.public GSE277078 GSE298812 ...   (or no argument: every dataset below)

Each matrix is streamed from GEO and only the panel CpGs are kept, so a 3 GB file never
lands on disk: the result is a small parquet in data/public/<GSE>.parquet (CpG × samples,
beta values), next to the sample descriptions from the series matrix (<GSE>_samples.csv).
"""

from __future__ import annotations

import gzip
import io
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

from burotu.config import DATA_DIR

OUT = DATA_DIR / "public"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}/{kind}/{name}"

# gse -> (file, role, short description). File None = beta values from the series matrix itself.
DATASETS = {
    "GSE277078": ("GSE277078_matrix_processed.txt.gz", "plasma", "Lung cancer plasma cfDNA, EPIC (185)"),
    "GSE298812": ("GSE298812_processed_data.txt.gz", "plasma", "Liver cancer, cirrhosis, fibrosis plasma, EPIC (245)"),
    "GSE129374": (None, "plasma", "Liver cancer vs cirrhosis plasma, 450K (44)"),
    "GSE108462": ("GSE108462_Normalised_matrix.csv.gz", "plasma", "Metastatic prostate cancer plasma, 450K (181)"),
    "GSE55763": ("GSE55763_normalized_betas.txt.gz", "blood", "Healthy population blood, 450K (2,711)"),
    "GSE87571": (
        ["GSE87571_matrix1of2.txt.gz", "GSE87571_matrix2of2.txt.gz"],
        "blood",
        "Blood, ages 14-94, 450K (732)",
    ),
}

SKIP_COLUMN = re.compile(r"det|pval|p\.val|p_val", re.I)


def _url(gse: str, name: str, kind: str = "suppl") -> str:
    return GEO.format(stub=gse[:-3] + "nnn", gse=gse, kind=kind, name=name)


def _open(url: str) -> io.TextIOBase:
    resp = urllib.request.urlopen(url, timeout=120)
    return io.TextIOWrapper(gzip.GzipFile(fileobj=resp), encoding="utf-8", errors="replace")


def series_samples(gse: str) -> tuple[pd.DataFrame, io.TextIOBase]:
    """Sample descriptions from the series matrix; returns the open stream at the table start."""
    stream = _open(_url(gse, f"{gse}_series_matrix.txt.gz", "matrix"))
    rows: dict[str, list[str]] = {}
    for line in stream:
        if line.startswith("!series_matrix_table_begin"):
            break
        if line.startswith("!Sample_"):
            parts = [p.strip('"') for p in line.rstrip("\n").split("\t")]
            key = parts[0][8:]
            while key in rows:
                key += "+"
            rows[key] = parts[1:]
    meta = pd.DataFrame(rows)
    return meta, stream


def _filter(stream: io.TextIOBase, panel: set[str], sep: str) -> pd.DataFrame:
    header = [h.strip('"') for h in stream.readline().rstrip("\n").split(sep)]
    keep = [i for i, h in enumerate(header) if i > 0 and h and not SKIP_COLUMN.search(h)]
    ids, values = [], []
    for line in stream:
        cpg = line[: line.find(sep)].strip('"')
        if cpg in panel:
            parts = line.rstrip("\n").split(sep)
            ids.append(cpg)
            values.append([parts[i].strip('"') for i in keep])
        if line.startswith("!series_matrix_table_end"):
            break
    df = pd.DataFrame(values, index=ids, columns=[header[i] for i in keep])
    return df.apply(pd.to_numeric, errors="coerce").astype(np.float32)


def fetch(gse: str, panel: list[str]) -> pd.DataFrame:
    """Download, cut to the panel, save; returns CpG × samples beta values."""
    OUT.mkdir(parents=True, exist_ok=True)
    files, _role, _desc = DATASETS[gse]
    meta, stream = series_samples(gse)
    meta.to_csv(OUT / f"{gse}_samples.csv", index=False)
    wanted = set(panel)
    if files is None:
        beta = _filter(stream, wanted, "\t")
    else:
        stream.close()
        parts = []
        for name in [files] if isinstance(files, str) else files:
            sep = "," if name.endswith(".csv.gz") else "\t"
            parts.append(_filter(_open(_url(gse, name)), wanted, sep))
        beta = pd.concat(parts, axis=1)
    beta = beta.reindex(panel)
    # Some matrices put an unnamed detection p-value column after each sample ("X1", "X1.1").
    pvals = [c for c in beta.columns if c.endswith(".1") and c[:-2] in beta.columns]
    beta = beta.drop(columns=pvals)
    if beta.max().max() > 1.5:  # some matrices are in percent
        beta = beta / 100
    beta.to_parquet(OUT / f"{gse}.parquet")
    return beta


def main(argv: list[str]) -> None:
    import json

    from burotu.analysis.engines import ASSETS

    panel = json.loads((ASSETS / "baseline" / "meta.json").read_text())["panel"]
    for gse in argv or list(DATASETS):
        beta = fetch(gse, panel)
        print(
            f"{gse}: {beta.shape[1]} samples, {beta.notna().any(axis=1).sum()} of {len(panel)} panel CpGs", flush=True
        )


if __name__ == "__main__":
    main(sys.argv[1:])


BLOOD_DONORS = ("GSE55763", "GSE87571")


def extra_blood_donors(panel: list[str], fill: pd.Series) -> pd.DataFrame:
    """Healthy-population blood from the open GEO cohorts (panel × donors), gaps filled with `fill`."""
    parts = []
    for gse in BLOOD_DONORS:
        b = pd.read_parquet(OUT / f"{gse}.parquet").reindex(panel)
        b.columns = [f"{gse}:{c}" for c in b.columns]
        parts.append(b)
    blood = pd.concat(parts, axis=1)
    blood = blood.loc[:, blood.notna().mean() > 0.9]  # drop failed arrays
    return blood.apply(lambda col: col.fillna(fill)).clip(0, 1).astype(np.float32)

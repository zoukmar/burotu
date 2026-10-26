"""Download GEO data: whole blood (the cfDNA background) and real cfDNA (validation)."""

from __future__ import annotations

import gzip
import re
import shutil

import httpx
import pandas as pd

from burotu.config import BLOOD_GEO_URL, HEALTHY, MOSS_CFDNA_URL, RAW_DIR

# Moss et al. sample title fragment -> label used across Burotu.
MOSS_LABELS = [
    ("lung_cancer", "Lung"),
    ("colon_cancer", "Colorectal"),
    ("breast_cancer", "Breast"),
    ("CUP", "Unknown primary"),
]


def _download(url: str, dest) -> None:
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, timeout=600, follow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    shutil.move(tmp, dest)


def load_blood() -> pd.DataFrame:
    """Whole-blood profiles of 164 healthy adults (Hannum 2013, GSE40279), CpG × individuals."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / "blood_hannum.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    gz = RAW_DIR / "GSE40279_subset.txt.gz"
    _download(BLOOD_GEO_URL, gz)
    df = pd.read_csv(gz, sep="\t", index_col=0, dtype={0: str})
    df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
    df.index.name = "cpg_id"
    df.to_parquet(cache)
    gz.unlink()
    return df


def moss_label(title: str) -> str | None:
    """Map a Moss et al. sample title to a Burotu label (None = sample not used)."""
    if re.match(r"cfDNA_healthy_individual|cfDNA_(young|old)_(male|female)_pool", title):
        return HEALTHY
    for key, label in MOSS_LABELS:
        if key in title:
            return label
    return None


def load_moss_cfdna() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real plasma cfDNA (Moss et al. 2018, GSE122126, EPIC array).

    Returns (beta matrix CpG × samples, metadata with a simplified label). Only cfDNA
    from healthy people and from cancer patients (lung, colon, breast, unknown
    primary) is kept.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / "moss_cfdna.parquet"
    meta_path = RAW_DIR / "moss_cfdna_meta.csv"
    if cache.exists() and meta_path.exists():
        meta = pd.read_csv(meta_path)
        meta["label"] = meta["title"].map(moss_label)  # labels always follow the current code
        return pd.read_parquet(cache), meta

    gz = RAW_DIR / "GSE122126_EPIC.txt.gz"
    _download(MOSS_CFDNA_URL, gz)
    titles, gsms = [], []
    with gzip.open(gz, "rt") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                titles = [t.strip('"') for t in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                gsms = [t.strip('"') for t in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                break
        df = pd.read_csv(f, sep="\t", index_col=0, comment="!")
    df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
    df.index.name = "cpg_id"

    meta = pd.DataFrame({"gsm": gsms, "title": titles})
    meta["label"] = meta["title"].map(moss_label)
    meta = meta.dropna(subset=["label"]).reset_index(drop=True)
    df = df[meta["gsm"]]
    df.to_parquet(cache)
    meta.to_csv(meta_path, index=False)
    gz.unlink()
    return df, meta


if __name__ == "__main__":
    b = load_blood()
    print("blood", b.shape)
    m, md = load_moss_cfdna()
    print("moss", m.shape)
    print(md["label"].value_counts())

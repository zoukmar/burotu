"""Téléchargement des données GEO : sang total (fond cfDNA) et cfDNA réel (validation)."""

from __future__ import annotations

import gzip
import re
import shutil

import httpx
import pandas as pd

from burotu.config import BLOOD_GEO_URL, MOSS_CFDNA_URL, RAW_DIR


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
    """Profils de sang total de 164 adultes sains (Hannum 2013, GSE40279), CpG × individus."""
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


def load_moss_cfdna() -> tuple[pd.DataFrame, pd.DataFrame]:
    """cfDNA plasmatique réel (Moss et al. 2018, GSE122126, EPIC).

    Renvoie (matrice bêta CpG × échantillons, métadonnées avec un label simplifié).
    Seuls les échantillons cfDNA de sujets sains et de patients atteints de cancer
    (poumon, côlon, sein, primitif inconnu) sont conservés.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / "moss_cfdna.parquet"
    meta_path = RAW_DIR / "moss_cfdna_meta.csv"
    if cache.exists() and meta_path.exists():
        return pd.read_parquet(cache), pd.read_csv(meta_path)

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

    def label(t: str) -> str | None:
        if re.match(r"cfDNA_healthy_individual|cfDNA_(young|old)_(male|female)_pool", t):
            return "Sain"
        for key, lab in [
            ("lung_cancer", "Poumon"),
            ("colon_cancer", "Colorectal"),
            ("breast_cancer", "Sein"),
            ("CUP", "Primitif inconnu"),
        ]:
            if key in t:
                return lab
        return None

    meta = pd.DataFrame({"gsm": gsms, "title": titles})
    meta["label"] = meta["title"].map(label)
    meta = meta.dropna(subset=["label"]).reset_index(drop=True)
    df = df[meta["gsm"]]
    df.to_parquet(cache)
    meta.to_csv(meta_path, index=False)
    gz.unlink()
    return df, meta


if __name__ == "__main__":
    b = load_blood()
    print("sang", b.shape)
    m, md = load_moss_cfdna()
    print("moss", m.shape)
    print(md["label"].value_counts())

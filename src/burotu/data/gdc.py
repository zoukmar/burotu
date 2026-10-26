"""Téléchargement des profils de méthylation tumoraux TCGA (GDC, accès ouvert).

Chaque fichier GDC « Methylation Beta Value » (450K, pipeline SeSAMe) contient
deux colonnes sans en-tête : identifiant de sonde CpG et valeur bêta (0 à 1).
"""

from __future__ import annotations

import io
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import numpy as np
import pandas as pd

from burotu.config import CANCER_CLASSES, RAW_DIR, SEED, TUMORS_PER_CLASS

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data/"


def list_tumor_files(projects: list[str]) -> list[dict]:
    """Liste les fichiers bêta 450K de tumeurs primaires pour des projets TCGA."""
    filt = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "data_type", "value": "Methylation Beta Value"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
            {"op": "=", "content": {"field": "platform", "value": "Illumina Human Methylation 450"}},
            {"op": "=", "content": {"field": "cases.samples.sample_type", "value": "Primary Tumor"}},
            {"op": "in", "content": {"field": "cases.project.project_id", "value": projects}},
        ],
    }
    fields = "file_id,file_name,cases.case_id,cases.project.project_id"
    r = httpx.post(GDC_FILES, json={"filters": filt, "fields": fields, "size": 5000}, timeout=120)
    r.raise_for_status()
    out = []
    for h in r.json()["data"]["hits"]:
        case = h["cases"][0]
        out.append({"file_id": h["file_id"], "case_id": case["case_id"], "project": case["project"]["project_id"]})
    return out


def select_files(per_class: int = TUMORS_PER_CLASS, seed: int = SEED) -> pd.DataFrame:
    """Tire `per_class` tumeurs par classe CSO, une seule par patient."""
    rng = random.Random(seed)
    rows = []
    for cls, projects in CANCER_CLASSES.items():
        files = list_tumor_files(projects)
        seen, uniq = set(), []
        for f in sorted(files, key=lambda x: x["file_id"]):
            if f["case_id"] not in seen:
                seen.add(f["case_id"])
                uniq.append(f)
        rng.shuffle(uniq)
        for f in uniq[:per_class]:
            rows.append({**f, "cancer_class": cls})
    return pd.DataFrame(rows)


def _fetch_beta(file_id: str, retries: int = 4) -> pd.Series:
    for attempt in range(retries):
        try:
            r = httpx.get(GDC_DATA + file_id, timeout=300, follow_redirects=True)
            r.raise_for_status()
            s = pd.read_csv(io.BytesIO(r.content), sep="\t", header=None, index_col=0).iloc[:, 0]
            return s.astype("float32")
        except Exception:  # réseau instable : on réessaie
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def download_tumors(per_class: int = TUMORS_PER_CLASS, workers: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Télécharge les tumeurs et renvoie (matrice bêta CpG × échantillons, métadonnées).

    Le résultat est mis en cache dans data/raw/tcga_tumors.npy (lu en memmap).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / "tcga_tumors.npy"
    meta_path = RAW_DIR / "tcga_tumors_meta.csv"
    part_dir = RAW_DIR / "tcga_parts"
    probes_path = part_dir / "_probes.txt"
    if cache.exists() and meta_path.exists():
        return _load_cache(cache, probes_path, meta_path)

    meta = select_files(per_class)
    # Chaque fichier est aligné sur un index de sondes commun et stocké en
    # float32 sur disque : la mémoire reste faible et le téléchargement reprend
    # là où il s'est arrêté en cas de coupure.
    part_dir.mkdir(exist_ok=True)
    probes: pd.Index | None = pd.Index(probes_path.read_text().split()) if probes_path.exists() else None

    todo = [fid for fid in meta["file_id"] if not (part_dir / f"{fid}.npy").exists()]
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_fetch_beta, fid): fid for fid in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            fid = futs[fut]
            try:
                s = fut.result()
                if probes is None:
                    probes = pd.Index(sorted(s.index))
                    probes_path.write_text("\n".join(probes))
                np.save(part_dir / f"{fid}.npy", s.reindex(probes).to_numpy(np.float32))
                del s
            except Exception as e:  # on garde le reste du lot
                print(f"  échec {fid}: {e}")
            if i % 20 == 0 or i == len(futs):
                print(f"  TCGA {i}/{len(futs)}", flush=True)

    done = [fid for fid in meta["file_id"] if (part_dir / f"{fid}.npy").exists()]
    meta = meta[meta["file_id"].isin(done)].reset_index(drop=True)
    arr = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32, shape=(len(probes), len(done)))
    for j, fid in enumerate(meta["file_id"]):
        arr[:, j] = np.load(part_dir / f"{fid}.npy")
    arr.flush()
    del arr
    meta.to_csv(meta_path, index=False)
    return _load_cache(cache, probes_path, meta_path)


def _load_cache(cache, probes_path, meta_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge la matrice en mémoire partagée (memmap), sans copie."""
    meta = pd.read_csv(meta_path)
    arr = np.load(cache, mmap_mode="r")
    probes = pd.Index(probes_path.read_text().split(), name="cpg_id")
    return pd.DataFrame(arr, index=probes, columns=meta["file_id"].tolist(), copy=False), meta


if __name__ == "__main__":
    m, md = download_tumors()
    print(m.shape)
    print(md["cancer_class"].value_counts())

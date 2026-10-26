"""Format d'échange entre le laboratoire partenaire et Burotu.

Le labo renvoie, pour chaque échantillon, un fichier TSV de comptages de lectures
bisulfite par CpG du panel ciblé (équivalent d'un fichier de couverture Bismark
restreint au panel) :

    # burotu-bmc v1
    # sample_id=BRT-000123
    # lab=Labo partenaire
    # assay=targeted-bisulfite-panel-v1
    cpg_id    meth    total
    cg00001   3       812
    ...

`meth` = lectures méthylées sur ce CpG, `total` = profondeur de lecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MAGIC = "# burotu-bmc v1"


@dataclass
class LabSample:
    sample_id: str
    counts: pd.DataFrame  # index cpg_id, colonnes meth, total
    header: dict[str, str] = field(default_factory=dict)

    @property
    def beta(self) -> pd.Series:
        total = self.counts["total"].replace(0, np.nan)
        return (self.counts["meth"] / total).astype("float32")


def write_lab_file(path: str | Path, sample_id: str, cpg_ids, meth, total, **header) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [MAGIC, f"# sample_id={sample_id}"] + [f"# {k}={v}" for k, v in header.items()]
    df = pd.DataFrame({"cpg_id": list(cpg_ids), "meth": np.asarray(meth, int), "total": np.asarray(total, int)})
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        df.to_csv(f, sep="\t", index=False)
    return path


class LabFormatError(ValueError):
    pass


def read_lab_file(path: str | Path) -> LabSample:
    path = Path(path)
    header: dict[str, str] = {}
    with open(path) as f:
        first = f.readline().strip()
        if first != MAGIC:
            raise LabFormatError(f"{path.name}: en-tête '{MAGIC}' manquant")
        pos = f.tell()
        line = f.readline()
        while line.startswith("#"):
            k, _, v = line[1:].strip().partition("=")
            header[k.strip()] = v.strip()
            pos = f.tell()
            line = f.readline()
        f.seek(pos)
        df = pd.read_csv(f, sep="\t")
    missing = {"cpg_id", "meth", "total"} - set(df.columns)
    if missing:
        raise LabFormatError(f"{path.name}: colonnes manquantes {sorted(missing)}")
    if "sample_id" not in header:
        raise LabFormatError(f"{path.name}: sample_id absent de l'en-tête")
    if (df["meth"] < 0).any() or (df["meth"] > df["total"]).any():
        raise LabFormatError(f"{path.name}: comptages incohérents (meth > total ou négatif)")
    df = df.drop_duplicates("cpg_id").set_index("cpg_id")[["meth", "total"]]
    return LabSample(sample_id=header["sample_id"], counts=df, header=header)

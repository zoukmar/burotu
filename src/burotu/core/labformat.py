"""Exchange format between the partner lab and Burotu ("burotu-bmc").

For each sample the lab returns a TSV file of bisulfite read counts per CpG of the
targeted panel (equivalent to a Bismark coverage file restricted to the panel):

    # burotu-bmc v1
    # sample_id=BRT-000123
    # lab=Partner lab
    # assay=targeted-bisulfite-panel-v1
    cpg_id    meth    total
    cg00001   3       812
    ...

`meth` = methylated reads at this CpG, `total` = read depth.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MAGIC = "# burotu-bmc v1"


@dataclass
class LabSample:
    sample_id: str
    counts: pd.DataFrame  # index cpg_id, columns meth and total
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
    return parse_lab_text(path.read_text(), path.name)


def parse_lab_text(text: str, name: str = "lab file") -> LabSample:
    """Parse a burotu-bmc file from its text (an upload, an API payload or a file on disk)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != MAGIC:
        raise LabFormatError(f"{name}: missing '{MAGIC}' header")
    header: dict[str, str] = {}
    i = 1
    while i < len(lines) and lines[i].startswith("#"):
        k, _, v = lines[i][1:].strip().partition("=")
        header[k.strip()] = v.strip()
        i += 1
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[i:])), sep="\t")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise LabFormatError(f"{name}: unreadable table ({e})") from e
    missing = {"cpg_id", "meth", "total"} - set(df.columns)
    if missing:
        raise LabFormatError(f"{name}: missing columns {sorted(missing)}")
    if "sample_id" not in header:
        raise LabFormatError(f"{name}: no sample_id in header")
    for col in ("meth", "total"):
        if not pd.api.types.is_integer_dtype(df[col]):
            raise LabFormatError(f"{name}: '{col}' must contain whole read counts")
    if (df["meth"] < 0).any() or (df["meth"] > df["total"]).any():
        raise LabFormatError(f"{name}: inconsistent counts (meth > total or negative)")
    df = df.drop_duplicates("cpg_id").set_index("cpg_id")[["meth", "total"]]
    return LabSample(sample_id=header["sample_id"], counts=df, header=header)

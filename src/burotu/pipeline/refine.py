"""Pipeline step 2 (refine): raw downloads -> Burotu format, ready for training.

    python -m burotu.pipeline.refine --store D:\\Burotu\\Data              # everything, resumable
    python -m burotu.pipeline.refine --store ... --only gdc --budget 170   # one family, time-boxed
    python -m burotu.pipeline.refine --store ... --status                  # what is done / left

Every raw dataset becomes one folder with the same three files, whatever its origin:

    <store>/refined/<version>/sources/<origin>/<dataset>/
        beta.parquet   panel CpGs x samples, beta values (float32, NaN = not measured)
        samples.csv    one row per sample, standard columns (SAMPLE_COLUMNS)
        info.json      source files, sample count, panel coverage, method, date

plus <store>/refined/<version>/catalog.csv (every sample of every source) and status.json.

The script only needs the standard library, numpy, pandas and pyarrow (isal speeds up gzip if
installed), and imports nothing from Burotu: it can be copied next to the data and run there.
Work is cut into units (one GDC batch, one GEO matrix, one Loyfer sample); a unit is written as
soon as it is done, so a run stopped at any point resumes where it left off.
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import gzip
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "v2"
SAMPLE_COLUMNS = [
    "sample_id",
    "dataset",
    "origin",
    "role",
    "tissue",
    "subtype",
    "condition",
    "age",
    "sex",
    "platform",
    "coverage",
    "raw_title",
]
# role: tumor | normal (non-cancer tissue) | blood | cell (sorted / purified cell type) | model (organoid, cell line)
#       | plasma | unknown


class Run:
    def __init__(self, store: Path, budget: float):
        self.store = store
        self.out = store / "refined" / VERSION
        self.end = time.time() + budget
        panel_file = self.out / "panel.csv"
        if not panel_file.exists():
            shutil.copy(store / "refined" / "v1" / "processed" / "panel.csv", panel_file)
        self.panel = pd.read_csv(panel_file).iloc[:, 0].tolist()
        self.pos = {c: i for i, c in enumerate(self.panel)}

    def left(self) -> float:
        return self.end - time.time()

    def src(self, origin: str, dataset: str) -> Path:
        d = self.out / "sources" / origin / dataset
        d.mkdir(parents=True, exist_ok=True)
        return d


def _write(run: Run, origin: str, dataset: str, beta: pd.DataFrame, samples: pd.DataFrame, info: dict) -> None:
    d = run.src(origin, dataset)
    beta = beta.reindex(run.panel).astype(np.float32)
    samples = samples.copy()
    samples["dataset"], samples["origin"] = dataset, origin
    samples["coverage"] = samples["sample_id"].map(beta.notna().mean()).round(3)
    for c in SAMPLE_COLUMNS:
        if c not in samples:
            samples[c] = None
    beta.to_parquet(d / "beta.parquet")
    samples[SAMPLE_COLUMNS].to_csv(d / "samples.csv", index=False)
    info = {
        "dataset": dataset,
        "origin": origin,
        "samples": int(beta.shape[1]),
        "panel_cpgs_with_data": int(beta.notna().any(axis=1).sum()),
        "median_coverage": float(samples["coverage"].median()) if len(samples) else 0.0,
        "refined_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "version": VERSION,
        **info,
    }
    (d / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(
        f"  wrote {origin}/{dataset}: {beta.shape[1]} samples, median coverage {info['median_coverage']:.2f}",
        flush=True,
    )


def _status(run: Run, key: str, state: str, note: str = "") -> None:
    f = run.out / "status.json"
    s = json.loads(f.read_text()) if f.exists() else {}
    s[key] = {"state": state, "note": note, "at": datetime.datetime.now().isoformat(timespec="seconds")}
    f.write_text(json.dumps(dict(sorted(s.items())), indent=1, ensure_ascii=False))


def _gunzip(path: Path) -> subprocess.Popen:
    """Decompress with ISA-L when available (3x faster than gzip), as a separate process (second core)."""
    try:
        import isal  # noqa: F401

        cmd = [sys.executable, "-m", "isal.igzip", "-d", "-c", str(path)]
    except ImportError:
        cmd = ["gzip", "-dc", str(path)]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 20)


def gz_ok(path: Path) -> bool:
    """Cheap truncation check: a complete gzip ends with the size of its content (mod 2^32)."""
    return subprocess.run(["gzip", "-t", str(path)], capture_output=True).returncode == 0


# ---------------------------------------------------------------------------------------------
# GDC: one file per sample (cg id <tab> beta), metadata from GDC/manifest.csv
# ---------------------------------------------------------------------------------------------
GDC_ROLE = {
    "Primary Tumor": "tumor",
    "Metastatic": "tumor",
    "Recurrent Tumor": "tumor",
    "Additional Metastatic": "tumor",
    "Additional - New Primary": "tumor",
    "Primary Blood Derived Cancer - Bone Marrow": "tumor",
    "Primary Blood Derived Cancer - Peripheral Blood": "tumor",
    "Recurrent Blood Derived Cancer - Bone Marrow": "tumor",
    "Recurrent Blood Derived Cancer - Peripheral Blood": "tumor",
    "FFPE Scrolls": "tumor",
    "FFPE Recurrent": "tumor",
    "Solid Tissue Normal": "normal",
    "Bone Marrow Normal": "normal",
    "Blood Derived Normal": "blood",
    "Lymphoid Normal": "normal",
    "Fibroblasts from Bone Marrow Normal": "cell",
    "Next Generation Cancer Model": "model",
    "Expanded Next Generation Cancer Model": "model",
    "Human Tumor Original Cells": "model",
}


def _gdc_patterns(run: Run) -> Path:
    f = run.out / "work" / "gdc_patterns.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():  # "cg123<TAB>" and EPIC v2 "cg123_TC21<TAB>"
        f.write_text("".join(f"{c}\t\n{c}_\n" for c in run.panel))
    return f


def refine_gdc(run: Run) -> None:
    root = run.store / "GDC"
    manifest = pd.read_csv(root / "manifest.csv").drop_duplicates("file_name").set_index("file_name")
    patterns = _gdc_patterns(run)
    for proj_dir in sorted(p for p in root.iterdir() if (p / "raw").is_dir()):
        proj = proj_dir.name
        work = run.out / "work" / "gdc" / proj
        work.mkdir(parents=True, exist_ok=True)
        done = set()
        for p in work.glob("part-*.parquet"):
            done.update(pd.read_parquet(p).columns)
        files = sorted(f.name for f in (proj_dir / "raw").iterdir() if f.name.endswith(".txt"))
        todo = [f for f in files if manifest.at[f, "file_id"] not in done] if files else []
        todo = [f for f in todo if f in manifest.index]
        batch_no = len(list(work.glob("part-*.parquet")))
        fresh = not (run.out / "sources" / "GDC" / proj / "info.json").exists()
        while todo and run.left() > 25:
            batch, todo = todo[:150], todo[150:]
            try:
                out = subprocess.run(
                    ["grep", "-H", "-F", "-f", str(patterns), *[str(proj_dir / "raw" / f) for f in batch]],
                    capture_output=True,
                    timeout=min(run.left(), 3600),
                ).stdout.decode()
            except subprocess.TimeoutExpired:
                return  # out of time: this batch is redone next run
            cols: dict[str, dict[str, float]] = {manifest.at[f, "file_id"]: {} for f in batch}
            for line in out.splitlines():
                path, _, rest = line.partition(":")
                cpg, _, val = rest.partition("\t")
                fid = manifest.at[Path(path).name, "file_id"]
                v = float(val) if val not in ("NA", "") else np.nan
                cols[fid].setdefault(cpg.split("_")[0], v)  # EPIC v2 replicates: keep the first
            beta = pd.DataFrame(cols).reindex(run.panel).astype(np.float32)
            beta.to_parquet(work / f"part-{batch_no:04d}.parquet")
            batch_no += 1
            fresh = True
            done.update(beta.columns)
        left = len([f for f in files if f in manifest.index and manifest.at[f, "file_id"] not in done])
        if done and fresh:
            beta = pd.concat([pd.read_parquet(p) for p in sorted(work.glob("part-*.parquet"))], axis=1)
            m = manifest.reset_index().set_index("file_id").loc[list(beta.columns)].reset_index()
            by_tissue = m["tissue_type"].map({"Normal": "normal", "Tumor": "tumor"}).fillna("unknown")
            samples = pd.DataFrame(
                {
                    "sample_id": m["file_id"],
                    "role": m["sample_type"].map(GDC_ROLE).fillna(by_tissue),
                    "tissue": m["primary_site"],
                    "subtype": m["primary_diagnosis"].fillna(m["disease_type"]),
                    "condition": m["sample_type"].fillna(m["tissue_type"]),
                    "age": m["age"],
                    "sex": m["sex"].astype("string").str[0].str.upper(),
                    "platform": m["platform"],
                    "raw_title": m["project"].astype(str) + " " + m["sample"].fillna(m["case"]).astype(str),
                }
            )
            _write(
                run,
                "GDC",
                proj,
                beta,
                samples,
                {
                    "source": f"GDC/{proj}/raw ({len(done)} files)",
                    "method": "Sesame level-3 betas cut to the panel (EPIC v2 replicate probes: first kept)",
                    "files_missing_from_download": int(len(manifest[manifest.project == proj]) - len(files)),
                },
            )
        state = "done" if not left else "partial"
        _status(run, f"GDC/{proj}", state, f"{len(done)} samples refined, {left} files left in this run")
        if run.left() <= 25:
            return


# ---------------------------------------------------------------------------------------------
# GEO: large processed matrices (CpG rows, one beta column per sample, optional p-value columns)
# ---------------------------------------------------------------------------------------------
def _series(path: Path) -> pd.DataFrame:
    rows: dict[str, list[str]] = {}
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_"):
                parts = [p.strip('"') for p in line.rstrip("\n").split("\t")]
                key = parts[0][8:]
                if key == "characteristics_ch1" and parts[1] and ":" in parts[1]:
                    key = "ch:" + parts[1].split(":")[0].strip().lower()
                    parts = [parts[0]] + [p.split(":", 1)[1].strip() if ":" in p else p for p in parts[1:]]
                while key in rows:
                    key += "+"
                rows[key] = parts[1:]
    return pd.DataFrame(rows)


def _matrix(run: Run, path: Path) -> pd.DataFrame:
    sep = "," if ".csv" in path.name else "\t"
    proc = _gunzip(path)
    header = [h.strip('"') for h in proc.stdout.readline().decode().rstrip("\r\n").split(sep)]
    keep = [i for i, h in enumerate(header) if i > 0 and h and not re.search(r"pval|p\.val|det", h, re.I)]
    wanted = set(run.panel)
    ids, vals = [], []
    for line in proc.stdout:
        end = line.find(sep.encode())
        cpg = line[:end].strip(b'"').decode()
        if cpg in wanted:
            parts = line.decode().rstrip("\r\n").split(sep)
            ids.append(cpg)
            vals.append([parts[i].strip('"') for i in keep])
            if len(ids) == len(wanted):
                break
    proc.kill()
    df = pd.DataFrame(vals, index=ids, columns=[header[i] for i in keep]).apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated()]
    if df.max().max() > 1.5:
        df = df / 100
    return df.astype(np.float32)


def _capper_samples(meta: pd.DataFrame, columns, key: str) -> pd.DataFrame:
    """Heidelberg brain-tumour sets: the matrix says "SAMPLE n", the series matrix gives n in title/description."""
    if key == "title":
        n = meta["title"].str.extract(r"sample (\d+)", flags=re.I)[0]
    else:
        n = meta["description"].str.extract(r"SAMPLE (\d+)")[0]
    meta = meta.assign(col="SAMPLE " + n).set_index("col").reindex(list(columns))
    cls = meta.get("ch:methylation class", pd.Series(index=meta.index, dtype=object))
    return pd.DataFrame(
        {
            "sample_id": list(columns),
            "geo": meta["geo_accession"].values,
            "role": np.where(cls.str.contains("CONTR", na=False), "normal", "tumor"),
            "tissue": "Brain",
            "subtype": cls.values,
            "condition": meta.get("ch:material"),
            "platform": "450K",
            "raw_title": meta["title"].values,
        }
    )


GEO_MATRICES = {
    # dataset: (matrix files, sample-label builder, description)
    "GSE90496": (["GSE90496_beta.txt.gz"], "capper_title", "Capper 2018 brain tumours, reference set (450K)"),
    "GSE109379": (
        ["GSE109379_processed_data.txt.gz"],
        "capper_desc",
        "Capper 2018 brain tumours, validation set (450K)",
    ),
    "GSE140686": (
        ["GSE140686_GPL13534_matrix_processed.txt.gz", "GSE140686_GPL21145_matrix_processed.txt.gz"],
        "plain",
        "Koelsche 2021 sarcomas (450K + EPIC)",
    ),
    "GSE237299": (
        ["GSE237299_GPL13534_beta.csv.gz", "GSE237299_GPL23976_beta.csv.gz"],
        "sentrix",
        "Haematolymphoid tumours (450K + EPIC)",
    ),
    # Normal tissues and sorted blood cells: the series matrices carry the betas.
    "GSE50192": (["GSE50192_series_matrix.txt.gz"], "series", "Lokk 2014, 17 normal tissues (450K)"),
    "GSE35069": (["GSE35069_series_matrix.txt.gz"], "series", "Reinius 2012, sorted blood cells (450K)"),
    "GSE110554": (["GSE110554_series_matrix.txt.gz"], "series", "Salas 2018, sorted blood cells (EPIC)"),
    "GSE167998": (["GSE167998_matrix_processed.txt.gz"], "series", "Salas 2022, 12 sorted blood cell types (EPIC)"),
    "GSE97466": (["GSE97466_series_matrix.txt.gz"], "series", "Thyroid tumours, benign and malignant (450K)"),
}


def _series_samples(meta: pd.DataFrame, columns, gse: str) -> pd.DataFrame:
    meta = meta.set_index("geo_accession", drop=False)
    if not set(columns) & set(meta.index):  # matrix columns are titles, not GSM ids
        meta = meta.set_index("title", drop=False)
    meta = meta.reindex(list(columns))
    ch = {c[3:]: meta[c] for c in meta.columns if c.startswith("ch:")}
    text = (
        meta.get("source_name_ch1", "").fillna("") + " " + ch.get("tissue", ch.get("cell type", "")).fillna("")
    ).str.lower()
    role = {"GSE50192": "normal", "GSE35069": "cell", "GSE110554": "cell", "GSE167998": "cell"}.get(gse)
    if role is None:
        role = np.where(text.str.contains("normal|benign|adenoma|goiter|hyperplas"), "normal", "tumor")
    tissue = ch.get("tissue", ch.get("cell type", meta.get("source_name_ch1")))
    return pd.DataFrame(
        {
            "sample_id": list(columns),
            "role": role,
            "tissue": getattr(tissue, "values", tissue),
            "subtype": (
                ch.get("cell type") if ch.get("cell type") is not None else ch.get("diagnosis", ch.get("histology"))
            ),
            "age": pd.to_numeric(ch.get("age", pd.Series(dtype=float)), errors="coerce").values
            if "age" in ch
            else None,
            "sex": ch["gender"].str[0].str.upper().values
            if "gender" in ch
            else (ch["sex"].str[0].str.upper().values if "sex" in ch else None),
            "raw_title": meta["title"].values,
        }
    )


def refine_geo(run: Run) -> None:
    for gse, (files, how, desc) in GEO_MATRICES.items():
        d = run.out / "sources" / "GEO" / gse
        if (d / "info.json").exists():
            continue
        raw = run.store / "GEO" / gse / "raw"
        missing = [f for f in files if not (raw / f).exists()]
        if missing:
            _status(run, f"GEO/{gse}", "blocked", f"missing download: {', '.join(missing)}")
            continue
        size = sum((raw / f).stat().st_size for f in files)
        if run.left() < 30 + size / 120e6:  # ~120 MB/s of gzip on the workstation VM
            _status(run, f"GEO/{gse}", "todo", "waiting for a run with enough time")
            continue
        broken = [f for f in files if size < 2e9 and not gz_ok(raw / f)]
        if broken:
            _status(run, f"GEO/{gse}", "blocked", f"incomplete download, re-download: {', '.join(broken)}")
            continue
        work = run.out / "work" / "geo" / gse
        work.mkdir(parents=True, exist_ok=True)
        parts = []
        for f in files:
            part = work / (f.split(".")[0] + ".parquet")
            if not part.exists():
                if run.left() < 30 + (raw / f).stat().st_size / 120e6:
                    break
                t = time.time()
                df = _matrix(run, raw / f)
                if len(df) < 0.05 * len(run.panel):
                    _status(run, f"GEO/{gse}", "blocked", f"{f}: only {len(df)} panel CpGs found (truncated file?)")
                    break
                df.to_parquet(part)
                print(f"  {gse}/{f}: {df.shape[1]} samples in {time.time() - t:.0f}s", flush=True)
            parts.append(pd.read_parquet(part))
        if len(parts) < len(files):
            if json.loads((run.out / "status.json").read_text()).get(f"GEO/{gse}", {}).get("state") != "blocked":
                _status(run, f"GEO/{gse}", "partial", f"{len(parts)}/{len(files)} matrices")
            continue
        beta = pd.concat(parts, axis=1)
        beta = beta.loc[:, ~beta.columns.duplicated()]
        if how.startswith("capper"):
            meta = _series(raw / f"{gse}_series_matrix.txt.gz")
            samples = _capper_samples(meta, beta.columns, "title" if how == "capper_title" else "desc")
        elif how == "sentrix":
            desc_t = pd.read_csv(
                raw / "GSE237299_all_sample_descriptions_unedited.txt.gz", sep="\t", dtype=str, encoding="latin-1"
            )
            desc_t["col"] = desc_t["idat file 1"].str.replace("_Grn.idat.gz", "", regex=False)
            m = desc_t.drop_duplicates("col").set_index("col").reindex(list(beta.columns))
            samples = pd.DataFrame(
                {
                    "sample_id": list(beta.columns),
                    "role": "tumor",
                    "tissue": m["source name"].values,
                    "subtype": m["title"].values,
                    "condition": m["Material"].values,
                    "age": pd.to_numeric(m["Age"], errors="coerce").values,
                    "sex": m["Sex"].values,
                    "platform": m["platform"].values,
                    "raw_title": m["description"].values,
                }
            )
        elif how == "series":
            meta = (
                _series(raw / f"{gse}_series_matrix.txt.gz") if (raw / f"{gse}_series_matrix.txt.gz").exists() else None
            )
            samples = (
                _series_samples(meta, beta.columns, gse)
                if meta is not None
                else pd.DataFrame({"sample_id": list(beta.columns), "role": "unknown"})
            )
        else:
            samples = pd.DataFrame(
                {
                    "sample_id": list(beta.columns),
                    "role": "tumor",
                    "tissue": "Soft tissue / bone",
                    "raw_title": list(beta.columns),
                }
            )
        if "geo" in samples:  # name samples by their GSM id when the matrix uses its own labels
            gsm = samples["geo"].fillna(samples["sample_id"])
            beta.columns, samples["sample_id"] = list(gsm), list(gsm)
        _write(
            run,
            "GEO",
            gse,
            beta,
            samples,
            {
                "source": f"GEO/{gse}/raw: {', '.join(files)}",
                "description": desc,
                "method": "processed betas cut to the panel",
            },
        )
        _status(run, f"GEO/{gse}", "done", desc)


# ---------------------------------------------------------------------------------------------
# Loyfer 2023 cell-type atlas (GSE186458): whole-genome bisulfite, wgbstools .beta inside one tar
# ---------------------------------------------------------------------------------------------
def _loyfer_ranges(run: Run, raw: Path) -> np.ndarray:
    """CpG-index range [start, end) of the homogeneous block holding each panel CpG (hg38)."""
    f = run.out / "work" / "loyfer" / "panel_blocks.npy"
    if f.exists():
        return np.load(f)
    coords = pd.read_csv(run.out / "panel_hg38.csv").set_index("cpg_id").reindex(run.panel)
    blocks = pd.read_csv(
        raw / "GSE186458_blocks.s207.hg38.bed.gz", sep="\t", header=None, names=["chr", "start", "end", "s", "e"]
    )
    ranges = np.zeros((len(run.panel), 2), dtype=np.int64)
    for chrom, b in blocks.groupby("chr"):
        starts, ends = b["start"].to_numpy(), b["end"].to_numpy()
        s, e = b["s"].to_numpy(), b["e"].to_numpy()
        for i, (c, pos) in enumerate(zip(coords["CpG_chrm"], coords["CpG_beg"], strict=True)):
            if c != chrom or pd.isna(pos):
                continue
            j = bisect.bisect_right(starts, pos) - 1
            if j >= 0 and pos <= ends[j]:
                ranges[i] = (s[j], e[j])
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, ranges)
    return ranges


def refine_loyfer(run: Run) -> None:
    gse = "GSE186458"
    raw = run.store / "GEO" / gse / "raw"
    d = run.out / "sources" / "GEO" / gse
    if (d / "info.json").exists():
        return
    work = run.out / "work" / "loyfer"
    work.mkdir(parents=True, exist_ok=True)
    index_file = work / "tar_index.json"
    if not index_file.exists():
        index = {}
        with tarfile.open(raw / f"{gse}_RAW.tar") as tar:
            for m in tar:
                if m.name.endswith(".hg38.beta"):
                    index[m.name] = [m.offset_data, m.size]
        index_file.write_text(json.dumps(index))
    index = json.loads(index_file.read_text())
    ranges = _loyfer_ranges(run, raw)
    ok = ranges[:, 1] > ranges[:, 0]
    with open(raw / f"{gse}_RAW.tar", "rb") as tar:
        for name, (offset, size) in sorted(index.items()):
            out = work / (name.split("_")[0] + ".npy")
            if out.exists():
                continue
            if run.left() < 15:
                break
            tar.seek(offset)
            a = np.frombuffer(tar.read(size), dtype=np.uint8).reshape(-1, 2).astype(np.int64)
            cm, cc = np.concatenate([[0], np.cumsum(a[:, 0])]), np.concatenate([[0], np.cumsum(a[:, 1])])
            s, e = ranges[:, 0] - 1, ranges[:, 1] - 1  # wgbstools CpG indices are 1-based
            s, e = np.clip(s, 0, len(a)), np.clip(e, 0, len(a))
            meth, cov = cm[e] - cm[s], cc[e] - cc[s]
            beta = np.where(ok & (cov >= 5), meth / np.maximum(cov, 1), np.nan).astype(np.float32)
            np.save(out, beta)
    done = sorted(work.glob("GSM*.npy"))
    if len(done) < len(index):
        _status(run, f"GEO/{gse}", "partial", f"{len(done)}/{len(index)} samples")
        return
    beta = pd.DataFrame({p.stem: np.load(p) for p in done}, index=run.panel)
    meta = _series(raw / f"{gse}_series_matrix.txt.gz").set_index("geo_accession").reindex(beta.columns)
    samples = pd.DataFrame(
        {
            "sample_id": list(beta.columns),
            "role": "cell",
            "tissue": meta["ch:tissue"].values,
            "subtype": meta["ch:cell type"].values,
            "condition": meta["description"].values,
            "age": pd.to_numeric(meta["ch:age"], errors="coerce").values,
            "sex": meta["ch:sex"].values,
            "platform": "WGBS",
            "raw_title": meta["title"].values,
        }
    )
    _write(
        run,
        "GEO",
        gse,
        beta,
        samples,
        {
            "source": f"GEO/{gse}/raw/{gse}_RAW.tar (*.hg38.beta)",
            "description": "Loyfer 2023 methylation atlas of 39 purified cell types (WGBS)",
            "method": "each panel CpG = methylated / covered reads over its Loyfer homogeneous block (s207, hg38), "
            "NaN under 5 reads; positions from the Zhou InfiniumAnnotation HM450 hg38 manifest",
            "panel_cpgs_in_a_block": int(ok.sum()),
        },
    )
    _status(run, f"GEO/{gse}", "done", f"{len(done)} samples")


# ---------------------------------------------------------------------------------------------
# Already refined: the 18 GEO panel exports and the v1 training data
# ---------------------------------------------------------------------------------------------
def refine_exports(run: Run) -> None:
    for d in sorted((run.store / "GEO").glob("*/panel")):
        gse = d.parent.name
        if (run.out / "sources" / "GEO" / gse / "info.json").exists() or run.left() < 20:
            continue
        info = json.loads((d / "info.json").read_text())
        beta = pd.read_csv(d / "beta.tsv.gz", sep="\t", index_col=0)
        s = pd.read_csv(d / "samples.csv")
        role = s["sample_type"].map({"blood": "blood", "plasma": "plasma", "tissue": "normal", "tumor": "tumor"})
        role = role.where(
            ~(s["sample_type"].eq("tissue") & s["condition"].str.contains("tumou?r|cancer", case=False, na=False)),
            "tumor",
        ).fillna("unknown")
        samples = pd.DataFrame(
            {
                "sample_id": s["sample_id"],
                "role": role,
                "tissue": s.get("condition"),
                "subtype": s.get("cancer_type"),
                "condition": s.get("condition"),
                "age": s.get("age"),
                "sex": s.get("sex"),
                "raw_title": s.get("raw_title"),
            }
        )
        _write(
            run,
            "GEO",
            gse,
            beta,
            samples,
            {
                "source": f"GEO/{gse}/panel",
                "description": info.get("description"),
                "method": "panel export of October 29 (format change only)",
            },
        )
        _status(run, f"GEO/{gse}", "done", info.get("description", ""))
    v1 = run.store / "refined" / "v1" / "processed"
    if not (run.out / "sources" / "GDC" / "TCGA" / "info.json").exists() and (v1 / "tumor_train.parquet").exists():
        beta = pd.concat(
            [pd.read_parquet(v1 / "tumor_train.parquet"), pd.read_parquet(v1 / "tumor_test.parquet")], axis=1
        )
        m = pd.read_csv(v1 / "tumor_meta.csv").set_index("file_id").reindex(beta.columns)
        samples = pd.DataFrame(
            {
                "sample_id": list(beta.columns),
                "role": "tumor",
                "tissue": m["cancer_class"].values,
                "subtype": m["project"].values,
                "condition": "split " + m["split"].astype(str).values,
                "platform": "450K",
                "raw_title": m["project"].values,
            }
        )
        _write(
            run,
            "GDC",
            "TCGA",
            beta,
            samples,
            {
                "source": "refined/v1/processed/tumor_*.parquet",
                "description": "TCGA tumours used by the v1 models (48 per class)",
            },
        )
        _status(run, "GDC/TCGA", "done", "v1 training tumours")
        blood = pd.concat(
            [pd.read_parquet(v1 / "blood_train.parquet"), pd.read_parquet(v1 / "blood_test.parquet")], axis=1
        )
        samples = pd.DataFrame(
            {
                "sample_id": list(blood.columns),
                "role": "blood",
                "tissue": "Whole blood",
                "platform": "450K",
                "raw_title": "Hannum 2013",
            }
        )
        _write(
            run,
            "GEO",
            "GSE40279",
            blood,
            samples,
            {
                "source": "refined/v1/processed/blood_*.parquet",
                "description": "Hannum 2013 whole blood (v1 background)",
            },
        )
        _status(run, "GEO/GSE40279", "done", "v1 blood background")


NOT_YET = {
    "figshare/SPOT-MAS_Nguyen2023": "region-level cfDNA sequencing: needs a region -> panel mapping (planned)",
    "Zenodo/Nassiri2020_brain_cfMeDIP": "cfMeDIP windows (enrichment, not betas): kept for later",
    "Zenodo/Kang2017_CancerLocator": "code + features of a published model: reference only",
    "GEO/GSE149438": "targeted sequencing: needs coordinates -> panel (planned)",
    "GEO/GSE93203": "targeted sequencing: needs coordinates -> panel (planned)",
    "GEO/GSE79279": "targeted sequencing: needs coordinates -> panel (planned)",
    "GEO/GSE63775": "targeted sequencing: needs coordinates -> panel (planned)",
}


def catalog(run: Run) -> pd.DataFrame:
    parts = [pd.read_csv(f) for f in sorted((run.out / "sources").glob("*/*/samples.csv"))]
    cat = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SAMPLE_COLUMNS)
    cat.to_csv(run.out / "catalog.csv", index=False)
    return cat


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--budget", type=float, default=1e9, help="stop starting new work after this many seconds")
    ap.add_argument("--only", default="exports,gdc,geo,loyfer")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    run = Run(a.store, a.budget)
    if not a.status:
        for key, note in NOT_YET.items():
            _status(run, key, "not yet", note)
        steps = {"exports": refine_exports, "gdc": refine_gdc, "geo": refine_geo, "loyfer": refine_loyfer}
        for name in a.only.split(","):
            if run.left() > 15:
                print(f"[{name}]", flush=True)
                steps[name](run)
    cat = catalog(run)
    s = json.loads((run.out / "status.json").read_text()) if (run.out / "status.json").exists() else {}
    print(f"\n{len(cat)} samples refined in {cat['dataset'].nunique() if len(cat) else 0} datasets")
    if len(cat):
        print(cat.groupby("role").size().to_string())
    for k, v in s.items():
        print(f"  {v['state']:9} {k:38} {v['note']}")


if __name__ == "__main__":
    main()

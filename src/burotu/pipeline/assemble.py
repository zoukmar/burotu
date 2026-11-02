"""Pipeline step 2b (assemble): refined sources -> the training reference of a refined version.

    python -m burotu.pipeline assemble            # writes <store>/refined/v2/processed/

Reads `<store>/refined/v2/sources/*/*` (written by burotu.pipeline.refine) and builds the same
Reference files as v1 (panel, blood_train/test, tumor_train/test, tumor_meta), so every model
trains on v2 unchanged with BUROTU_REFINED=v2. What changes from v1:

  * tumours: every refined tumour with >= 80 % of the panel measured, from TCGA, CPTAC, TARGET,
    CCDI, HCMI, CGCI and the GEO brain / sarcoma / blood-cancer series, mapped to tissue-of-origin
    classes (CLASS_RULES): the 20 v1 classes plus ovary, bile duct, eye and blood cancers;
    at most MAX_PER_CLASS per class, drawn evenly across source datasets;
  * healthy blood: about 6,000 donors from 9 open cohorts instead of 164 (Hannum);
  * the healthy background also receives DNA from endothelium, erythroid progenitors and
    hepatocytes (Loyfer atlas), as real healthy plasma does (Moss 2018);
  * train / test split by sample within each source (30 % test), donors likewise.

The panel stays the v1 panel (3,148 CpGs): the refined sources only hold those sites.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

from burotu.config import SEED, STORE
from burotu.data.reference import Reference, _impute_rows, split_ids

VERSION = "v2"
MIN_COVERAGE = 0.8
MAX_PER_CLASS = 300
MIN_PER_CLASS = 30
TEST_FRAC = 0.3

# Healthy whole blood (no cancer): population cohorts, ageing, twins, and non-cancer disease
# controls (rheumatoid arthritis, IBD) so that inflammation is part of "healthy" too.
BLOOD_SETS = ["GSE40279", "GSE55763", "GSE87571", "GSE56105", "GSE72680", "GSE72775", "GSE42861",
              "GSE234190"]  # fmt: skip
# Non-blood DNA in healthy plasma: component -> (Loyfer cell types, maximum share in a sample).
PLASMA_COMPONENTS = {"endothelium": ("Endothelium", 0.12), "erythroid": ("Erythrocyte progenitors", 0.25),
                     "hepatocyte": ("Hepatocyte", 0.08)}  # fmt: skip

# Tissue of origin: first matching rule wins; dataset rules first, then the sample's tissue text.
DATASET_CLASS = {"TCGA-CHOL": "Bile duct", "TCGA-UVM": "Eye", "TCGA-OV": "Ovary", "TCGA-UCS": "Uterus",
                 "TCGA-ACC": "Adrenal", "TCGA-PCPG": "Adrenal", "TCGA-KICH": "Kidney", "TARGET-NBL": "Adrenal",
                 "TARGET-WT": "Kidney", "TARGET-CCSK": "Kidney", "TCGA-DLBC": "Blood cancer",
                 "TARGET-AML": "Blood cancer", "TARGET-ALL-P3": "Blood cancer", "GSE237299": "Blood cancer",
                 "GSE90496": "Brain", "GSE140686": "Soft tissue", "TARGET-OS": "Soft tissue",
                 "CGCI-HTMCP-CC": "Cervix"}  # fmt: skip
CLASS_RULES = [
    ("Bile duct", r"bile|gallbladder"), ("Liver", r"liver"), ("Eye", r"\beye\b"), ("Ovary", r"ovary"),
    ("Lung", r"lung|bronch"), ("Colorectal", r"colon|rect|colorectal"), ("Breast", r"breast"),
    ("Pancreas", r"pancrea"), ("Oesophagus-Stomach", r"stomach|esophag|oesophag"),
    ("Head and neck", r"larynx|tongue|mouth|pharynx|head and neck|\blip\b|tonsil|gum|palate|salivary|parotid|nasal"),
    ("Prostate", r"prostate"), ("Kidney", r"kidney"), ("Bladder", r"bladder"),
    ("Uterus", r"uter|endometri"), ("Cervix", r"cervix"), ("Thyroid", r"thyroid"), ("Skin", r"skin"),
    ("Brain", r"brain|spinal cord|central nervous|cerebr"), ("Mesothelium", r"mesothel|pleura"),
    ("Adrenal", r"adrenal"), ("Testis", r"testi"), ("Thymus", r"thymus"),
    ("Blood cancer", r"hematopoietic|bone marrow|blood|lymph node|spleen|^pb$"),
    ("Soft tissue", r"soft tissue|connective|bone"),
]  # fmt: skip


def tumour_class(dataset: str, tissue) -> str | None:
    if dataset in DATASET_CLASS:
        return DATASET_CLASS[dataset]
    text = str(tissue).lower()
    if dataset == "TCGA":  # v1 tumours already carry their Burotu class
        return str(tissue)
    for cls, pattern in CLASS_RULES:
        if re.search(pattern, text):
            return cls
    return None


def _load(src, cat: pd.DataFrame) -> pd.DataFrame:
    """Panel × samples for the catalog rows `cat`, read source by source."""
    parts = []
    for (origin, dataset), rows in cat.groupby(["origin", "dataset"], sort=False):
        b = pd.read_parquet(src / origin / dataset / "beta.parquet", columns=list(rows["sample_id"]))
        b.columns = list(rows["uid"])
        parts.append(b)
    return pd.concat(parts, axis=1)[list(cat["uid"])]


# Each array cohort comes with its own lab batch (between-cohort spread of the per-site mean is
# about 3x the spread between people of one cohort). Every cohort is moved onto the reference
# batch (Hannum, the v1 background) site by site, keeping the differences between its people.
HARMONIZE_TO = "GSE40279"
# A targeted panel only works on sites that stay unmethylated in every healthy cell type that
# sheds DNA into plasma: sites above this mean beta in a plasma component are left out.
COMPONENT_MAX_BETA = 0.2


def assemble(
    store=STORE, seed: int = SEED, harmonize: bool = True, components: bool = False, panel_filter: bool = False
) -> Reference:
    root = store / "refined" / VERSION
    src = root / "sources"
    rng = np.random.default_rng(seed)
    panel = pd.read_csv(root / "panel.csv").iloc[:, 0].tolist()
    cat = pd.read_csv(root / "catalog.csv", low_memory=False)
    cat["sample_id"] = cat["sample_id"].astype(str)
    cat["uid"] = cat["dataset"] + ":" + cat["sample_id"]
    cat = cat.drop_duplicates("uid")
    good = cat["coverage"] >= MIN_COVERAGE

    # Healthy blood donors, split within each cohort.
    import os

    blood_sets = os.environ.get("BUROTU_ASSEMBLE_BLOOD", ",".join(BLOOD_SETS)).split(",")
    blood_cat = cat[good & cat["dataset"].isin(blood_sets) & cat["role"].eq("blood")]
    blood = _load(src, blood_cat)
    blood_med = pd.Series(np.nanmedian(blood.to_numpy(np.float32), axis=1), index=blood.index)
    blood_med = blood_med.fillna(float(np.nanmedian(blood_med)))
    blood = _impute_rows(blood, blood_med.to_numpy())
    cohort = pd.Series([u.split(":")[0] for u in blood.columns], index=blood.columns)
    if harmonize:
        target = blood.loc[:, cohort == HARMONIZE_TO].mean(axis=1)
        for _, cols in cohort.groupby(cohort).groups.items():
            shift = (target - blood[cols].mean(axis=1)).to_numpy()[:, None]
            blood[cols] = np.clip(blood[cols].to_numpy() + shift, 0, 1)
    # v1 test donors and tumours stay in test, so v2 models can still be scored on the v1 benchmark.
    v1 = store / "refined" / "v1" / "processed"
    v1_test = set("GSE40279:" + pd.read_parquet(v1 / "blood_test.parquet").columns)
    b_tr, b_te = [], []
    for _, rows in blood_cat.groupby("dataset"):
        tr, te = split_ids(rows["uid"], TEST_FRAC, rng)
        b_tr += [u for u in tr if u not in v1_test]
        b_te += te + [u for u in tr if u in v1_test]

    # Tumours, mapped to classes, capped per class evenly across source datasets.
    tum = cat[good & cat["role"].eq("tumor")].copy()
    tum["cancer_class"] = [tumour_class(d, t) for d, t in zip(tum["dataset"], tum["tissue"], strict=True)]
    tum = tum.dropna(subset=["cancer_class"])
    counts = tum["cancer_class"].value_counts()
    tum = tum[tum["cancer_class"].isin(counts.index[counts >= MIN_PER_CLASS])]
    picked = []
    for _, rows in tum.groupby("cancer_class"):
        rows = rows.sample(frac=1, random_state=seed)
        # round-robin over datasets so one large series cannot fill a class on its own
        rows = rows.assign(_r=rows.groupby("dataset").cumcount()).sort_values(["_r", "dataset"])
        picked.append(rows.head(MAX_PER_CLASS).drop(columns="_r"))
    tum = pd.concat(picked)
    tumors = _load(src, tum)
    parts = []
    for _, rows in tum.groupby("cancer_class"):
        sub = tumors[list(rows["uid"])]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(sub.to_numpy(np.float32), axis=1)
        med = np.where(np.isnan(med), blood_med.to_numpy(), med)
        parts.append(_impute_rows(sub, med))
    tumors = pd.concat(parts, axis=1)[list(tum["uid"])]
    tum["split"] = "train"
    for _, rows in tum.groupby("cancer_class"):
        _, te = split_ids(rows["uid"], TEST_FRAC, rng)
        tum.loc[tum["uid"].isin(te), "split"] = "test"
    v1_split = tum["dataset"].eq("TCGA")
    tum.loc[v1_split, "split"] = np.where(tum.loc[v1_split, "condition"].eq("split test"), "test", "train")
    meta = pd.DataFrame({"file_id": tum["uid"], "case_id": tum["sample_id"], "project": tum["dataset"],
                         "cancer_class": tum["cancer_class"], "split": tum["split"],
                         "subtype": tum["subtype"], "platform": tum["platform"]})  # fmt: skip
    t_tr = meta.loc[meta["split"] == "train", "file_id"].tolist()
    t_te = meta.loc[meta["split"] == "test", "file_id"].tolist()

    # Plasma components from the Loyfer atlas.
    lo = cat[cat["dataset"].eq("GSE186458")].copy()
    lo["component"] = None
    for comp, (cell, _) in PLASMA_COMPONENTS.items():
        lo.loc[lo["subtype"].astype(str).str.contains(cell, regex=False), "component"] = comp
    lo = lo.dropna(subset=["component"])
    cells = _impute_rows(_load(src, lo), blood_med.to_numpy())
    lo["split"] = "train"
    for _, rows in lo.groupby("component"):
        _, te = split_ids(rows["uid"], TEST_FRAC, rng)
        lo.loc[lo["uid"].isin(te), "split"] = "test"
    bmeta = pd.DataFrame({"sample_id": lo["uid"], "component": lo["component"], "split": lo["split"]})

    if panel_filter:
        comp_mean = pd.DataFrame({c: cells[list(r["sample_id"])].mean(axis=1) for c, r in bmeta.groupby("component")})
        raw = _load(src, lo).reindex(panel)  # before imputation: NaN = not measured in the atlas
        measured = raw.notna().mean(axis=1) > 0.5
        noisy = measured & (comp_mean.max(axis=1) > COMPONENT_MAX_BETA)
        panel = [c for c in panel if not noisy.get(c, False)]
    if not components:
        bmeta = bmeta.iloc[:0]
    settings = {
        "version": VERSION,
        "calibration_donors": 300 if len(b_tr) > 1000 else 35,
        "plasma_components": {k: v[1] for k, v in PLASMA_COMPONENTS.items()},
        "plasma_components_prob": 0.8,
        "classes": sorted(meta["cancer_class"].unique()),
        "blood_sets": BLOOD_SETS,
        "min_coverage": MIN_COVERAGE,
        "max_per_class": MAX_PER_CLASS,
        "harmonized_to": HARMONIZE_TO if harmonize else None,
        "panel_filter_component_max_beta": COMPONENT_MAX_BETA if panel_filter else None,
    }
    if not components:
        settings.pop("plasma_components")
    sel = lambda df, ids: df.loc[panel, ids].astype(np.float32)  # noqa: E731
    return Reference(
        panel=panel,
        blood_train=sel(blood, b_tr),
        blood_test=sel(blood, b_te),
        tumor_train=sel(tumors, t_tr),
        tumor_test=sel(tumors, t_te),
        tumor_meta=meta.reset_index(drop=True),
        background_train=sel(cells, bmeta.loc[bmeta["split"] == "train", "sample_id"].tolist()),
        background_test=sel(cells, bmeta.loc[bmeta["split"] == "test", "sample_id"].tolist()),
        background_meta=bmeta.reset_index(drop=True),
        settings=settings,
    )


def main(argv: list[str] | None = None) -> None:
    import os

    defaults = {"harmonize": "1", "components": "0", "panel_filter": "0"}
    flags = {k: os.environ.get(f"BUROTU_ASSEMBLE_{k.upper()}", v) == "1" for k, v in defaults.items()}
    ref = assemble(**flags)
    out = STORE / "refined" / os.environ.get("BUROTU_ASSEMBLE_OUT", VERSION) / "processed"
    ref.save(out)
    v1 = STORE / "refined" / "v1"
    # Real plasma used as checks only: same files as v1.
    shutil.copy(v1 / "processed" / "real_plasma_panel.parquet", out / "real_plasma_panel.parquet")
    if (v1 / "public").exists():
        shutil.copytree(v1 / "public", out.parent / "public", dirs_exist_ok=True)
    summary = {
        "panel": len(ref.panel),
        "blood_donors": {"train": ref.blood_train.shape[1], "test": ref.blood_test.shape[1]},
        "tumours": ref.tumor_meta.groupby(["cancer_class", "split"]).size().unstack().fillna(0).astype(int).to_dict(),
        "tumour_sources": ref.tumor_meta["project"].value_counts().to_dict(),
        "plasma_components": {
            f"{a}/{b}": int(n) for (a, b), n in ref.background_meta.groupby(["component", "split"]).size().items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"blood donors {ref.blood_train.shape[1]} train / {ref.blood_test.shape[1]} test")
    print(ref.tumor_meta.groupby(["cancer_class", "split"]).size().unstack())
    print(f"panel {len(ref.panel)} sites;", summary["plasma_components"])


if __name__ == "__main__":
    main(sys.argv[1:])

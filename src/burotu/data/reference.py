"""Build the reference: align sources, split train/test, design the targeted panel.

The panel mimics a targeted bisulfite sequencing panel: it keeps CpG sites that are
almost never methylated in blood (low background in plasma) and strongly methylated
in at least one tumour type. The panel is chosen on the training split only, so no
information leaks into the test set.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from burotu.config import PROCESSED_DIR, SEED


@dataclass
class Reference:
    panel: list[str]
    blood_train: pd.DataFrame  # panel × individuals
    blood_test: pd.DataFrame
    tumor_train: pd.DataFrame  # panel × tumours
    tumor_test: pd.DataFrame
    tumor_meta: pd.DataFrame  # file_id, cancer_class, split

    def save(self, out=PROCESSED_DIR) -> None:
        out.mkdir(parents=True, exist_ok=True)
        pd.Series(self.panel, name="cpg_id").to_csv(out / "panel.csv", index=False)
        for name in ("blood_train", "blood_test", "tumor_train", "tumor_test"):
            getattr(self, name).to_parquet(out / f"{name}.parquet")
        self.tumor_meta.to_csv(out / "tumor_meta.csv", index=False)

    @classmethod
    def load(cls, src=PROCESSED_DIR) -> Reference:
        return cls(
            panel=pd.read_csv(src / "panel.csv")["cpg_id"].tolist(),
            blood_train=pd.read_parquet(src / "blood_train.parquet"),
            blood_test=pd.read_parquet(src / "blood_test.parquet"),
            tumor_train=pd.read_parquet(src / "tumor_train.parquet"),
            tumor_test=pd.read_parquet(src / "tumor_test.parquet"),
            tumor_meta=pd.read_csv(src / "tumor_meta.csv"),
        )


def _impute_rows(df: pd.DataFrame, fill: np.ndarray) -> pd.DataFrame:
    """Replace the NaNs of each row with that row's `fill` value."""
    a = df.to_numpy(np.float32, copy=True)
    r, c = np.where(np.isnan(a))
    a[r, c] = fill[r]
    return pd.DataFrame(a, index=df.index, columns=df.columns)


def split_ids(ids, test_frac: float, rng: np.random.Generator):
    ids = np.array(sorted(ids))
    rng.shuffle(ids)
    n_test = max(1, round(len(ids) * test_frac))
    return list(ids[n_test:]), list(ids[:n_test])


def design_panel(
    blood: pd.DataFrame,
    tumors: pd.DataFrame,
    classes: pd.Series,
    per_class: int = 300,
    blood_max_mean: float = 0.10,
    blood_max_q95: float = 0.20,
) -> list[str]:
    """Select the panel CpG sites (training data only)."""
    b_mean = blood.mean(axis=1)
    b_q95 = blood.quantile(0.95, axis=1)
    quiet = b_mean.index[(b_mean < blood_max_mean) & (b_q95 < blood_max_q95)]
    t = tumors.loc[quiet]
    chosen: list[str] = []
    for cls in sorted(classes.unique()):
        cols = classes.index[classes == cls]
        # Score: median tumour methylation of the class minus the blood background,
        # plus a bonus for being specific to this class (helps tissue of origin).
        own = t[cols].median(axis=1)
        other = t.drop(columns=cols).median(axis=1)
        frac_hi = (t[cols] > 0.3).mean(axis=1)
        score = (own - b_mean.loc[quiet]) * frac_hi + 0.5 * (own - other)
        chosen.extend(score.sort_values(ascending=False).index[:per_class])
    # Pan-cancer markers: methylated in most tumours across all classes.
    pan = (t > 0.3).mean(axis=1).sort_values(ascending=False).index[:per_class]
    chosen.extend(pan)
    return list(dict.fromkeys(chosen))


def build_reference(
    blood: pd.DataFrame,
    tumors: pd.DataFrame,
    tumor_meta: pd.DataFrame,
    restrict_to: pd.Index | None = None,
    test_frac: float = 0.3,
    per_class: int = 300,
    seed: int = SEED,
) -> Reference:
    rng = np.random.default_rng(seed)
    common = blood.index.intersection(tumors.index)
    if restrict_to is not None:  # e.g. CpGs also on the EPIC array (Moss validation)
        common = common.intersection(restrict_to)
    blood = blood.loc[common]
    tumors = tumors.loc[common]

    # Missing values: median per CpG (blood) or per CpG and class (tumours).
    blood_med = np.nanmedian(blood.to_numpy(np.float32), axis=1)
    blood = _impute_rows(blood, blood_med)
    meta = pd.DataFrame({"file_id": list(tumors.columns)}).merge(tumor_meta, on="file_id", how="left")
    parts = []
    for _cls, grp in meta.groupby("cancer_class"):
        sub = tumors[grp["file_id"]]
        with warnings.catch_warnings():  # all-NaN rows fall back to the blood median
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(sub.to_numpy(np.float32), axis=1)
        med = np.where(np.isnan(med), blood_med, med)
        parts.append(_impute_rows(sub, med))
    tumors = pd.concat(parts, axis=1)[meta["file_id"].tolist()]

    b_tr, b_te = split_ids(blood.columns, test_frac, rng)
    meta["split"] = "train"
    for _cls, grp in meta.groupby("cancer_class"):
        _, te = split_ids(grp["file_id"], test_frac, rng)
        meta.loc[meta["file_id"].isin(te), "split"] = "test"
    t_tr = meta.loc[meta["split"] == "train", "file_id"].tolist()
    t_te = meta.loc[meta["split"] == "test", "file_id"].tolist()

    classes = meta.set_index("file_id").loc[t_tr, "cancer_class"]
    panel = design_panel(blood[b_tr], tumors[t_tr], classes, per_class=per_class)
    return Reference(
        panel=panel,
        blood_train=blood.loc[panel, b_tr],
        blood_test=blood.loc[panel, b_te],
        tumor_train=tumors.loc[panel, t_tr],
        tumor_test=tumors.loc[panel, t_te],
        tumor_meta=meta[["file_id", "case_id", "project", "cancer_class", "split"]],
    )


def main() -> None:
    from burotu.data.gdc import download_tumors
    from burotu.data.geo import load_blood, load_moss_cfdna

    tumors, tmeta = download_tumors()
    blood = load_blood()
    moss, _ = load_moss_cfdna()
    ref = build_reference(blood, tumors, tmeta, restrict_to=moss.index)
    ref.save()
    print(f"Panel: {len(ref.panel)} CpG")
    print(f"Blood: {ref.blood_train.shape[1]} train / {ref.blood_test.shape[1]} test")
    print(ref.tumor_meta.groupby(["cancer_class", "split"]).size().unstack())


if __name__ == "__main__":
    main()

"""Quality control of a lab file before any analysis.

A result is only as good as the sequencing behind it. These checks decide whether a
sample can be analysed, and their numbers are passed to Muse Spark, which explains
any problem in plain words.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from burotu.labformat import LabSample

MIN_COVERAGE = 0.95  # share of panel CpGs with at least one read
MIN_MEDIAN_DEPTH = 200
MAX_LOW_DEPTH_SHARE = 0.20  # share of panel CpGs read fewer than LOW_DEPTH times
LOW_DEPTH = 100


@dataclass
class QCReport:
    status: str  # "pass" | "warn" | "fail"
    panel_coverage: float
    median_depth: float
    p10_depth: float
    low_depth_share: float
    unknown_cpgs: int
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def check(sample: LabSample, panel: list[str]) -> QCReport:
    counts = sample.counts.reindex(panel)
    depth = counts["total"].fillna(0).to_numpy(dtype=float)
    covered = depth > 0
    coverage = float(covered.mean())
    median = float(np.median(depth[covered])) if covered.any() else 0.0
    p10 = float(np.percentile(depth[covered], 10)) if covered.any() else 0.0
    low = float((depth < LOW_DEPTH).mean())
    unknown = len(set(sample.counts.index) - set(panel))

    issues = []
    if coverage < MIN_COVERAGE:
        issues.append(f"Only {coverage:.0%} of the panel's CpG sites were read (expected at least {MIN_COVERAGE:.0%}).")
    if median < MIN_MEDIAN_DEPTH:
        issues.append(f"Median read depth is {median:.0f}, below the {MIN_MEDIAN_DEPTH} needed for reliable detection.")
    if low > MAX_LOW_DEPTH_SHARE:
        issues.append(f"{low:.0%} of CpG sites have fewer than {LOW_DEPTH} reads.")
    if unknown:
        issues.append(f"{unknown} CpG sites in the file are not part of the Burotu panel and were ignored.")

    if coverage < 0.8 or median < 100:
        status = "fail"
    elif any(not i.endswith("were ignored.") for i in issues):
        status = "warn"
    else:
        status = "pass"
    return QCReport(status, round(coverage, 4), round(median, 1), round(p10, 1), round(low, 4), unknown, issues)

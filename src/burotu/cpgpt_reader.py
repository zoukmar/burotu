"""CpGPT (Camillo et al., 2024), a DNA methylation foundation model, as a Burotu reader.

CpGPT is a transformer trained on more than 150,000 methylation arrays to rebuild masked
CpG values. Each CpG enters as its methylation value plus the Nucleotide Transformer v2
embedding of the 2,001 bases around it. We use the released `large` checkpoint (about
100M parameters, MIT licence, Hugging Face `lucascamillomd/cpgpt-models`) frozen, take its
sample embedding (the CLS token) for each plasma sample, and train two small heads on top
(burotu.model.heads): cancer signal, and tissue of origin.

The `cpgpt` package's own data pipeline loads the location tables of 324 species into
memory; this module instead keeps only the panel's 3,148 sites (`prepare_panel`), so the
reader runs in about 1 GB of memory next to Muse Glimmer.
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import numpy as np

MODEL_REPO = "lucascamillomd/cpgpt-models"
DEPS_REPO = "lucascamillomd/cpgpt-human-dependencies"
DNA_LLM = "nucleotide-transformer-v2-500m-multi-species"
CONTEXT = 2001
EMBED_DIM = 1024


def _human(db: Path) -> dict:
    with sqlite3.connect(db) as c:
        return pickle.loads(c.execute("select value from unnamed where key='homo_sapiens'").fetchone()[0])


def prepare_panel(panel: list[str], deps_dir: Path, out: Path) -> dict:
    """Write the panel's DNA embeddings and coordinates (CpGPT's chromosome vocabulary) to `out`."""
    probes = _human(deps_dir / "illumina_metadata.db")
    ens = _human(deps_dir / "ensembl_metadata.db")
    rows_of = ens[DNA_LLM][CONTEXT]
    vocab = ens["vocab"]
    keep, chroms, positions, rows = [], [], [], []
    for i, cpg in enumerate(panel):
        loc = probes.get(cpg)
        if loc is None or loc not in rows_of:
            continue
        chrom, pos = loc.split(":")
        keep.append(i)
        chroms.append(vocab[chrom])
        positions.append(int(pos))
        rows.append(rows_of[loc])
    mmap = np.memmap(
        deps_dir / "dna_embeddings" / "homo_sapiens" / DNA_LLM / f"{CONTEXT}bp_dna_embeddings.mmap",
        dtype="float32",
        mode="r",
        shape=(len(rows_of), EMBED_DIM),
    )
    keep, chroms, positions = np.array(keep), np.array(chroms), np.array(positions)
    order = np.lexsort((positions, chroms))  # CpGPT's "sorted_chromosome", chromosomes in a fixed order
    emb = np.asarray(mmap[np.array(rows)[order]], dtype=np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, panel_index=keep[order], chroms=chroms[order], positions=positions[order], dna=emb.astype(np.float16)
    )
    return {"panel": len(panel), "kept": len(keep)}


class CpGPTReader:
    """Frozen CpGPT: beta values on the panel -> one embedding per sample."""

    def __init__(self, config_path: Path, ckpt_path: Path, panel_assets: Path, device: str | None = None) -> None:
        import hydra
        import torch
        from cpgpt.model.components.legacy_numerics import enable_legacy_numerics
        from omegaconf import OmegaConf

        enable_legacy_numerics()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = OmegaConf.load(config_path)
        model = hydra.utils.instantiate(cfg.model)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict({k.replace("net._orig_mod.", "net."): v for k, v in state.items()}, strict=True)
        self.model = model.to(self.device).eval()
        a = np.load(panel_assets)
        self.index = a["panel_index"]
        self.chroms = torch.as_tensor(a["chroms"], dtype=torch.int32)
        self.positions = torch.as_tensor(a["positions"], dtype=torch.int32)
        self.dna = torch.as_tensor(a["dna"].astype(np.float32))

    def embed(self, beta: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """`beta`: samples × panel (NaN = not read). Returns samples × embedding."""
        torch = self.torch
        out = []
        for s in range(0, len(beta), batch_size):
            b = torch.as_tensor(beta[s : s + batch_size][:, self.index], dtype=torch.float32)
            n = len(b)
            batch = {
                "meth": b.to(self.device),
                "dna_embeddings": self.dna[None].expand(n, -1, -1).to(self.device),
                "chroms": self.chroms[None].expand(n, -1).to(self.device),
                "positions": self.positions[None].expand(n, -1).to(self.device),
                "obsm": None,
            }
            half = torch.float16 if self.device != "cpu" else torch.bfloat16
            with torch.no_grad(), torch.autocast(self.device, dtype=half):
                res = self.model._predict_forward_step(batch, 1)
            out.append(res["sample_embedding"].float().cpu().numpy())
        return np.concatenate(out)


def beta_from_counts(meth: np.ndarray, total: np.ndarray) -> np.ndarray:
    return np.where(total > 0, meth / np.maximum(total, 1), np.nan).astype(np.float32)

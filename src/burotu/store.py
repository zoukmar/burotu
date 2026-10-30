"""Move the data and weights store between the workstation, CI, the servers and Modal.

Git holds code only. Data and model weights live in the store (burotu.config.STORE):

    <store>/raw/<origin>/<dataset>/      downloads, never modified
    <store>/refined/<version>/          Burotu format, ready for training
    <store>/models/<model>/<version>/   weights + report of each trained run

The full store stays on the workstation (D:\\Burotu\\Data). What other machines need travels as
versioned bundles attached to GitHub Releases (not part of the Git history), pinned below:

    python -m burotu.store fetch models      # weights of the models in use (site, CI)
    python -m burotu.store fetch refined     # refined data for training (Modal, CI)
    python -m burotu.store pack models       # build a bundle from the local store
    python -m burotu.store publish models    # upload it (needs `gh` and a new pin below)
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from burotu.config import MODELS_STORE, REFINED_VERSION, STORE

REPO = "zoukmar/burotu"

# bundle -> (release tag, file name, sha256 or "" to trust the .sha256 file published with it).
PINS = {
    "models": (
        "store-models-2026-10-30",
        "burotu-models.tar.gz",
        "a698ba84a91ce6faabc7b4be67ce26a646a292dedfc5877b0829f5bb6f340162",
    ),
    "refined": (
        f"store-refined-{REFINED_VERSION}",
        f"burotu-refined-{REFINED_VERSION}.tar.gz",
        "58a939ad071a3b300849c8e6aff9e2ffcafa85adffb5966151fa510e494b5b9f",
    ),
}

# Paths inside the store that each bundle carries.
MODELS_IN_USE = ("classic", "logreg")


def _members(bundle: str) -> list[Path]:
    if bundle == "models":
        from burotu.models.registry import get

        return [Path("models") / m / get(m).version for m in MODELS_IN_USE]
    if bundle == "refined":
        base = Path("refined") / REFINED_VERSION
        # The benchmark keeps only the frozen sample lists: the arrays are rebuilt from a fixed seed.
        return [base / "processed", base / "public", base / "benchmark" / "calibration.csv",
                base / "benchmark" / "test.csv"]  # fmt: skip
    raise KeyError(bundle)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(bundle: str) -> Path:
    tag, name, _ = PINS[bundle]
    out = STORE / "dist" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for m in _members(bundle):
            if not (STORE / m).exists():
                raise FileNotFoundError(f"{STORE / m} is missing from the local store")
            tar.add(STORE / m, arcname=str(m))
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB) sha256 {sha256(out)} -> release {tag}")
    return out


def publish(bundle: str) -> None:
    tag = PINS[bundle][0]
    path = pack(bundle)
    exists = subprocess.run(["gh", "release", "view", tag, "-R", REPO], capture_output=True).returncode == 0
    if not exists:
        notes = f"Burotu store bundle ({bundle}). Data and weights live outside Git: see src/burotu/store.py."
        subprocess.run(["gh", "release", "create", tag, "-R", REPO, "--title", tag, "--notes", notes], check=True)
    subprocess.run(["gh", "release", "upload", tag, str(path), "-R", REPO, "--clobber"], check=True)


def fetch(bundle: str, force: bool = False) -> None:
    """Download a pinned bundle into the store (skipped when its contents are already there)."""
    tag, name, digest = PINS[bundle]
    if not force and all((STORE / m).exists() for m in _members(bundle)):
        print(f"{bundle}: already in {STORE}")
        return
    url = f"https://github.com/{REPO}/releases/download/{tag}/{name}"
    dest = STORE / "dist" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"{bundle}: downloading {url}")
    urllib.request.urlretrieve(url, dest)
    if not digest:  # published next to the bundle by the "Store bundles" workflow
        with urllib.request.urlopen(f"{url}.sha256") as r:
            digest = r.read().decode().strip()
    if sha256(dest) != digest:
        raise ValueError(f"{name}: checksum mismatch")
    with tarfile.open(dest) as tar:
        tar.extractall(STORE, filter="data")
    print(f"{bundle}: extracted into {STORE}")


def main(argv: list[str]) -> None:
    if len(argv) < 2 or argv[0] not in {"pack", "publish", "fetch"} or argv[1] not in PINS:
        sys.exit("usage: python -m burotu.store {pack|publish|fetch} {models|refined}")
    {"pack": pack, "publish": publish, "fetch": fetch}[argv[0]](argv[1])


if __name__ == "__main__":
    main(sys.argv[1:])

__all__ = ["MODELS_STORE", "fetch", "pack", "publish"]

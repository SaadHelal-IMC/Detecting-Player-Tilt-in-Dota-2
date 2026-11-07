"""Freeze the processed dataset as a fixed version for all later steps.

Copies the processed tables and the two interim files they depend on into
data/frozen/<version>/ and writes manifest.json with row counts, SHA-256
checksums of every file, the checksums of the scripts that produced them and
the git commit. Features, labels and models read only from a frozen version,
so rerunning collect or clean later cannot change the results unnoticed.

An existing version is never overwritten; a changed dataset gets a new one.
check(version) recomputes the checksums and fails if any file changed.

    python -m src.clean.freeze v1
    python -m src.clean.freeze v1 --check
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone

import pandas as pd

from src.collect import opendota as od

FILES = ["processed/matches.parquet", "processed/players.parquet",
         "processed/account_screen.parquet", "interim/candidates.csv",
         "interim/account_screen_review.csv"]
OPTIONAL = {"interim/account_screen_review.csv"}  # only there after a hand check
SCRIPTS = ["src/collect/constants.py", "src/collect/collect.py",
           "src/clean/clean.py", "src/clean/screen_accounts.py"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_dir(version):
    return od.DATA_DIR / "frozen" / version


def git_state():
    try:
        run = lambda *a: subprocess.run(["git", "--no-optional-locks", *a], cwd=od.REPO_ROOT,
                                        capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "uncommitted_changes": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def freeze(version):
    out = frozen_dir(version)
    if out.exists():
        raise SystemExit(f"{out} already exists, frozen versions are never overwritten")

    missing = [f for f in FILES if f not in OPTIONAL and not (od.DATA_DIR / f).exists()]
    if missing:
        raise SystemExit(f"missing {missing}, run clean and screen_accounts first")
    players = pd.read_parquet(od.DATA_DIR / "processed/players.parquet")
    if "suspect_account" not in players.columns:
        raise SystemExit("players.parquet has no suspect_account, run screen_accounts after clean")

    files = {}
    for f in FILES:
        src = od.DATA_DIR / f
        if not src.exists():
            continue
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        entry = {"sha256": sha256(dst), "bytes": dst.stat().st_size}
        if f.endswith(".parquet"):
            entry["rows"] = len(pd.read_parquet(dst))
        files[f] = entry

    manifest = {
        "version": version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "players": len(players),
        "matches": files["processed/matches.parquet"]["rows"],
        "suspect_accounts": int(players["suspect_account"].sum()),
        "files": files,
        "scripts": {s: sha256(od.REPO_ROOT / s) for s in SCRIPTS if (od.REPO_ROOT / s).exists()},
        "git": git_state(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"froze {version}: {manifest['players']} players, {manifest['matches']:,} matches, "
          f"{manifest['suspect_accounts']} suspect accounts\n  -> {out}")
    if manifest["git"]["uncommitted_changes"]:
        print("  note: the repo has uncommitted changes, the script checksums in the manifest are what counts")
    return manifest


def check(version):
    out = frozen_dir(version)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    changed = [f for f, e in manifest["files"].items()
               if not (out / f).exists() or sha256(out / f) != e["sha256"]]
    if changed:
        raise SystemExit(f"{version}: changed or missing files {changed}")
    print(f"{version}: all {len(manifest['files'])} files match the manifest")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check(a.version) if a.check else freeze(a.version)

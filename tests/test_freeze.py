"""Freeze step: copies, manifest, no overwrite, check catches a changed file."""
import json

import pandas as pd
import pytest

from src.clean import freeze as fz
from src.collect import opendota as od


@pytest.fixture
def data(monkeypatch, tmp_path):
    monkeypatch.setattr(od, "DATA_DIR", tmp_path / "data")
    proc, interim = tmp_path / "data" / "processed", tmp_path / "data" / "interim"
    proc.mkdir(parents=True)
    interim.mkdir(parents=True)
    pd.DataFrame({"account_id": [1, 1, 2], "match_id": [10, 11, 12]}).to_parquet(proc / "matches.parquet")
    pd.DataFrame({"account_id": [1, 2], "suspect_account": [False, True]}).to_parquet(proc / "players.parquet")
    pd.DataFrame({"account_id": [1, 2]}).to_parquet(proc / "account_screen.parquet")
    pd.DataFrame({"account_id": [1, 2], "status": "usable"}).to_csv(interim / "candidates.csv", index=False)
    return tmp_path / "data"


def test_freeze_and_check(data):
    m = fz.freeze("v1")
    out = data / "frozen" / "v1"
    assert m["players"] == 2 and m["matches"] == 3 and m["suspect_accounts"] == 1
    assert (out / "processed" / "matches.parquet").exists()
    assert "interim/account_screen_review.csv" not in m["files"]          # optional, not there yet
    assert json.loads((out / "manifest.json").read_text())["files"]["processed/players.parquet"]["rows"] == 2
    assert fz.check("v1")

    with pytest.raises(SystemExit):                                        # never overwritten
        fz.freeze("v1")

    pd.DataFrame({"account_id": [1]}).to_parquet(out / "processed" / "players.parquet")
    with pytest.raises(SystemExit):                                        # a changed file is caught
        fz.check("v1")


def test_freeze_needs_screen(data):
    pd.DataFrame({"account_id": [1, 2]}).to_parquet(data / "processed" / "players.parquet")
    with pytest.raises(SystemExit):
        fz.freeze("v1")

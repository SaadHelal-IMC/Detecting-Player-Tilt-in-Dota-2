"""Runs the day-1 check script end to end on fake API answers."""
import runpy
import sys

import pandas as pd

from src.collect import opendota as od


def fake_matches(n=30):
    rows = []
    t = 1_760_000_000
    for i in range(n):
        rows.append({
            "match_id": 8_000_000_000 + i, "player_slot": 0 if i % 2 else 130,
            "radiant_win": i % 3 == 0, "start_time": t, "duration": 2100,
            "game_mode": 22, "lobby_type": 7, "hero_id": 14 if i % 4 else 8,
            "kills": 5, "deaths": 6, "assists": 9, "gold_per_min": 450,
            "xp_per_min": 520, "last_hits": 160, "denies": 8, "hero_damage": None,
            "tower_damage": 900, "hero_healing": 0, "leaver_status": 0,
            "party_size": 1, "version": None,
        })  # average_rank left out on purpose
        t += 2100 + 600
    return rows


def test_check_fields_runs(monkeypatch, tmp_path, capsys):
    def fake_api_get(path, params=None, use_cache=True):
        if path.endswith("/matches"):
            assert params["lobby_type"] == 7 and "gold_per_min" in params["project"]
            return fake_matches()
        return {"profile": {"personaname": "tester"}, "rank_tier": 54}

    monkeypatch.setattr(od, "api_get", fake_api_get)
    monkeypatch.setattr(od, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_fields", "123"])
    runpy.run_module("src.collect.check_fields", run_name="__main__")

    out = capsys.readouterr().out
    assert "rank_tier=54 (Legend)" in out
    assert "30 ranked matches returned" in out
    assert "never returned: average_rank" in out
    assert "hero_damage" in out and "0.0%" in out
    saved = pd.read_csv(tmp_path / "interim" / "own_matches_check.csv")
    assert len(saved) == 30 and "win" in saved.columns

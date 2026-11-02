"""Runs the sampling probe on fake API answers."""
import runpy
import sys

from src.collect import opendota as od


def test_probe_sampling_runs(monkeypatch, capsys):
    def fake_api_get(path, params=None, use_cache=True):
        if path == "publicMatches":
            assert params == {"min_rank": 80, "max_rank": 85}
            return [{"match_id": 111, "start_time": 1_790_000_000, "avg_rank_tier": 81,
                     "num_rank_tier": 9, "lobby_type": 7, "game_mode": 22}]
        assert path == "matches/111"
        players = [{"account_id": None if i % 3 == 0 else 1000 + i, "rank_tier": 80 if i % 2 else None,
                    "personaname": f"p{i}", "player_slot": i, "hero_id": i + 1} for i in range(10)]
        return {"players": players}

    monkeypatch.setattr(od, "api_get", fake_api_get)
    monkeypatch.setattr(sys, "argv", ["probe", "80", "85"])
    runpy.run_module("src.collect.probe_sampling", run_name="__main__")

    out = capsys.readouterr().out
    assert "1 matches" in out
    assert "public account_id: 6/10, rank_tier shown: 5/10" in out

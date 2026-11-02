"""Clean step on a hand-built cache: each rule gets one player that triggers it."""
import json

import pandas as pd
import pytest

from src.clean import clean
from src.collect import opendota as od
from src.collect.constants import HISTORY_PARAMS, WINDOW_START_TS

DAY = 86400
T0 = 1_780_000_000


def match(i, t, avg_rank=32, radiant=True, win=True, duration=2000, lobby=7, party=1):
    return {
        "match_id": 7_000_000_000 + i, "player_slot": 0 if radiant else 128,
        "radiant_win": win if radiant else not win, "start_time": t, "duration": duration,
        "game_mode": 22, "lobby_type": lobby, "hero_id": 1 + i % 5, "kills": 5, "deaths": 5,
        "assists": 5, "gold_per_min": 400, "xp_per_min": 500, "last_hits": 150, "denies": 5,
        "hero_damage": 15000, "tower_damage": 1000, "hero_healing": 0, "leaver_status": 0,
        "party_size": party, "average_rank": avg_rank, "version": None,
    }


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(od, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(od, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(od, "REPO_ROOT", tmp_path)

    hist = {}
    # A: 150 good matches + 5 isolated old ones + 1 duplicate + 1 unranked + 1 remake
    a = [match(i, T0 + i * 3600, win=i % 2 == 0, radiant=i % 3 != 0) for i in range(150)]
    a += [match(1000 + i, T0 - 200 * DAY + i * 3600) for i in range(5)]
    a += [a[10], match(2000, T0 + 151 * 3600, lobby=0), match(2001, T0 + 152 * 3600, duration=300)]
    hist[1] = a[::-1]                                                  # API gives newest first
    hist[2] = [match(3000 + i, T0 + i * 3600) for i in range(80)]      # B: too few
    hist[5] = [match(5000 + i, T0 + i * 3600, avg_rank=None) for i in range(120)]  # E: no average_rank
    hist[6] = [match(6000 + i, T0 + i * 3600, avg_rank=74 if i % 2 else 76) for i in range(110)]  # F: Immortal, lobbies read Divine
    # G: 130 matches but only 90 inside the window -> dropped
    hist[7] = [match(7000 + i, 1_700_000_000 + i * 3600) for i in range(40)] + \
              [match(7100 + i, T0 + i * 3600) for i in range(90)]

    for acc, rows in hist.items():
        f = od.cache_file(f"players/{acc}/matches", HISTORY_PARAMS)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rows), encoding="utf-8")

    cands = pd.DataFrame([
        {"account_id": 1, "medal": 3, "status": "usable"},
        {"account_id": 2, "medal": 3, "status": "usable"},
        {"account_id": 3, "medal": 3, "status": "too_few"},
        {"account_id": 4, "medal": 4, "status": "usable"},  # D: not cached yet
        {"account_id": 5, "medal": 4, "status": "usable"},
        {"account_id": 6, "medal": 8, "status": "usable"},
        {"account_id": 7, "medal": 5, "status": "usable"},
    ])
    (tmp_path / "data" / "interim").mkdir(parents=True)
    cands.to_csv(tmp_path / "data" / "interim" / "candidates.csv", index=False)
    return tmp_path


def test_clean_rules(cache, capsys):
    df, players = clean.run()
    assert set(players["account_id"]) == {1, 5, 6}                   # B too few, C not usable, D not cached, G window
    assert (df["start_time"] >= WINDOW_START_TS).all()

    a = df[df["account_id"] == 1]
    assert len(a) == 151                                             # 150 + remake; old run, dup, unranked gone
    assert a["match_id"].is_unique
    assert (a["lobby_type"] == 7).all()
    assert a["start_time"].is_monotonic_increasing
    assert list(a["match_idx"]) == list(range(151))
    assert a["short_game"].sum() == 1
    assert (a["start_time"] >= T0).all()                             # 200-day-old games dropped

    first = a.iloc[0]                                                # i=0: dire (0 % 3 == 0), won
    assert not first["is_radiant"] and first["win"]
    assert a.iloc[1]["is_radiant"] and not a.iloc[1]["win"]

    p = players.set_index("account_id")
    assert p.loc[1, "bracket"] == 3
    assert p.loc[5, "bracket"] == 4 and pd.isna(p.loc[5, "lobby_medal"])
    assert p.loc[6, "bracket"] == 8 and p.loc[6, "lobby_medal"] == 7  # profile wins, lobby reads Divine
    assert (df.groupby("account_id")["bracket"].nunique() == 1).all()

    for f in ["data/processed/matches.parquet", "data/processed/players.parquet",
              "results/tables/dataset_summary.csv"]:
        assert (cache / f).exists()
    out = capsys.readouterr().out
    assert "history not cached yet: 1" in out

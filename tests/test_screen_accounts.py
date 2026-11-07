"""Account screen on hand-built players: a normal player, a 24/7 bot, a sold
account, two friends who 5-stack together, a player who only shares a seed,
a smurf and a player with one throw day."""
import numpy as np
import pandas as pd
import pytest

from src.clean import screen_accounts as sa
from src.collect import opendota as od

DAY, T0 = 86400, 1_760_000_000  # T0 = 2025-10-09 09:46 UTC
rng = np.random.default_rng(0)


def player(acc, times, heroes, steady=False, party=1, first_id=None, p_win=0.5):
    n = len(times)
    first_id = acc * 10_000 if first_id is None else first_id
    return pd.DataFrame({
        "account_id": acc, "match_id": np.arange(first_id, first_id + n), "start_time": times,
        "duration": 2400 if steady else rng.integers(1500, 3600, n),
        "hero_id": heroes, "kills": 5 if steady else rng.poisson(6, n),
        "deaths": 5 if steady else rng.poisson(6, n), "assists": 5 if steady else rng.poisson(10, n),
        "gold_per_min": 500.0 if steady else rng.normal(450, 120, n).clip(150), "party_size": party,
        "win": rng.random(n) < p_win,
    })


def throw_day_ts():
    return T0 - T0 % DAY + 70 * DAY


def evenings(days, hour=19):
    """3 matches a day at hour:00, :45 and +1:30 UTC."""
    day0 = T0 - T0 % DAY
    return np.array([day0 + d * DAY + hour * 3600 + k * 2700 for d in range(days) for k in range(3)])


@pytest.fixture
def data(monkeypatch, tmp_path):
    monkeypatch.setattr(od, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(od, "REPO_ROOT", tmp_path)

    human = player(1, evenings(100), rng.integers(1, 26, 300))
    bot = player(2, T0 + np.arange(300) * 2880, np.arange(300) % 3 + 1, steady=True)  # a game every 48 min, nonstop
    sold = player(3, np.r_[evenings(50), evenings(50, hour=7) + 50 * DAY],      # new owner: new heroes, mornings
                  np.r_[rng.integers(1, 21, 150), rng.integers(60, 81, 150)])
    f4 = player(4, evenings(70)[:200], rng.integers(1, 40, 200))
    f5 = player(5, evenings(70)[:200], rng.integers(1, 40, 200))
    f5.loc[:119, ["match_id", "start_time"]] = f4.loc[:119, ["match_id", "start_time"]].to_numpy()  # 120 games together
    f4.loc[:119, "party_size"] = f5.loc[:119, "party_size"] = 5
    p6 = player(6, evenings(40)[:120], rng.integers(1, 40, 120))
    p6.loc[0, ["match_id", "start_time"]] = f4.loc[150, ["match_id", "start_time"]].to_numpy()  # seed of 4 and 6
    smurf = player(7, evenings(60), rng.integers(1, 40, 180), p_win=0.85)
    throw_day = throw_day_ts()
    thrower = pd.concat([player(8, evenings(60), rng.integers(1, 40, 180)),
                         player(8, throw_day + 10 * 3600 + np.arange(10) * 1800, rng.integers(1, 40, 10), first_id=90_000)])
    thrower.loc[thrower["match_id"] >= 90_000, ["kills", "deaths", "assists", "win"]] = [0, 12, 0, False]  # 10 games fed and lost

    proc, interim = tmp_path / "data" / "processed", tmp_path / "data" / "interim"
    proc.mkdir(parents=True)
    interim.mkdir(parents=True)
    pd.concat([human, bot, sold, f4, f5, p6, smurf, thrower]).to_parquet(proc / "matches.parquet", index=False)
    pd.DataFrame({"account_id": range(1, 9), "bracket": [3, 8, 5, 2, 2, 4, 7, 6]}).to_parquet(proc / "players.parquet", index=False)
    seed = int(f4.loc[150, "match_id"])
    pd.DataFrame({"account_id": [4, 6, 9], "medal": 2, "seed_match_id": [seed, seed, 99],
                  "status": ["usable", "usable", "empty"]}).to_csv(interim / "candidates.csv", index=False)
    return tmp_path


def test_signals(data, capsys):
    s = sa.run().set_index("account_id")

    bot = s.loc[2]
    assert bot["quiet_share"] > 0.2                        # evenly spread play gives about 0.25
    assert bot["max_24h"] == 30                            # 24 h / 48 min
    assert bot["spread"] == 0 and bot["top3_share"] == 1
    assert bot["n_signals"] >= 4 and bot["suspect_account"]

    human = s.loc[1]
    assert human["quiet_share"] == 0 and human["max_24h"] == 3
    assert human["shift"] == 0                             # same hours all along
    assert human["n_signals"] == 0 and not human["suspect_account"]

    sold = s.loc[3]
    assert sold["shift"] == pytest.approx(1.0)             # heroes and hours both change completely
    assert sold["shift_at"] == pd.to_datetime(T0 - T0 % DAY + 50 * DAY + 7 * 3600, unit="s", utc=True)  # first morning game
    assert sold["n_signals"] == 1 and not sold["suspect_account"]   # one signal is not enough

    friend = s.loc[4]
    assert friend["shared_share"] == pytest.approx(120 / 200)    # the seed match does not count
    assert friend["stack5_share"] == pytest.approx(120 / 200)
    assert friend["shared_partner"] == 5 and friend["shared_max"] == 120
    assert friend["flag_stack5_share"] and friend["flag_shared_share"]
    assert friend["n_signals"] == 1                              # both are one party signal
    assert s.loc[6, "shared_share"] == 0 and s.loc[6, "shared_max"] == 0

    assert s.loc[4, "cv_group"] == s.loc[5, "cv_group"]          # 120 shared matches: same fold
    assert s.loc[6, "cv_group"] != s.loc[4, "cv_group"]          # one shared seed is not a link
    assert s["cv_group"].nunique() == 7

    smurf = s.loc[7]
    assert smurf["boost_wr"] > 0.75 and smurf["n_signals"] == 1
    assert smurf["suspect_account"]                              # boost alone is enough
    thrower = s.loc[8]
    assert thrower["throw_days"] == 1 and thrower["n_signals"] == 1
    assert thrower["first_throw_day"] == pd.to_datetime(throw_day_ts(), unit="s", utc=True)
    assert thrower["suspect_account"]                            # a throw day alone is enough
    assert human["boost_wr"] < 0.75 and human["throw_days"] == 0

    players = pd.read_parquet(data / "data" / "processed" / "players.parquet").set_index("account_id")
    assert players.loc[2, "suspect_account"] and not players.loc[1, "suspect_account"]
    assert players["cv_group"].notna().all()
    assert list(pd.read_csv(data / "data" / "interim" / "account_screen_check.csv")["account_id"]) == [2, 7, 8]
    summary = pd.read_csv(data / "results" / "tables" / "account_screen_summary.csv")
    assert "account_id" not in summary.columns             # no ids in the public repo
    assert summary["players"].sum() == 8 and summary["suspects"].sum() == 3

    sa.run()                                               # rerun must not duplicate columns in players.parquet
    assert pd.read_parquet(data / "data" / "processed" / "players.parquet").columns.is_unique


def test_review_overrides(data):
    review = data / "data" / "interim" / "account_screen_review.csv"
    pd.DataFrame({"account_id": [2, 4], "verdict": ["clear", " Suspect"], "note": ["", "farm partner"]}).to_csv(review, index=False)
    s = sa.run().set_index("account_id")
    assert not s.loc[2, "suspect_account"] and s.loc[2, "auto_suspect"]
    assert s.loc[4, "suspect_account"]

    pd.DataFrame({"account_id": [2], "verdict": ["maybe"]}).to_csv(review, index=False)
    with pytest.raises(SystemExit):
        sa.run()

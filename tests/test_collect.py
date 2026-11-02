"""Offline tests for the sampler/collector with a fake OpenDota."""
import pandas as pd
import pytest

from src.collect import collect
from src.collect import opendota as od
from src.collect.constants import HISTORY_PARAMS, WINDOW_START_TS

RECENT = 1_790_000_000  # 2026-09


class FakeOpenDota:
    """Two pages of public matches per medal band, 10 players per match.

    Ranks inside a match are mixed on purpose (one player one medal higher),
    some players are anonymous, and history length depends on the account id:
    ids ending in 3 have 40 ranked games, ending in 7 have none, others 150.
    """

    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after

    def __call__(self, path, params=None, use_cache=True):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise od.DailyLimitReached("fake quota used up")
        self.calls.append((path, dict(params or {})))
        od.calls_made += 1

        if path == "publicMatches":
            m = params["min_rank"] // 10
            page = 0 if "less_than_match_id" not in params else 1
            if page == 1 and params["less_than_match_id"] <= m * 1000 + 100:
                return []
            base = m * 1000 + (500 if page == 0 else 0)
            return [{"match_id": base + i, "lobby_type": 7 if i % 3 == 0 else 0} for i in range(1, 10)]

        if path.startswith("matches/"):
            mid = int(path.split("/")[1])
            m = mid // 1000
            players = []
            for i in range(10):
                acc = mid * 10 + i
                if i in (1, 5):
                    players.append({"account_id": None, "rank_tier": None})
                else:
                    med = min(m + 1, 8) if i == 9 else m
                    players.append({"account_id": acc, "rank_tier": med * 10 + 2})
            return {"players": players}

        if path.startswith("players/") and path.endswith("/matches"):
            assert params["lobby_type"] == 7 and params["limit"] == 500
            acc = int(path.split("/")[1])
            n = 40 if acc % 10 == 3 else 0 if acc % 10 == 7 else 150
            return [{"match_id": k, "start_time": RECENT - k * 3600} for k in range(n)]

        raise AssertionError(f"unexpected path {path}")


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(od, "DATA_DIR", tmp_path)
    monkeypatch.setattr(od, "RAW_DIR", tmp_path / "raw")
    return tmp_path


def load(data_dir):
    return pd.read_csv(data_dir / "interim" / "candidates.csv")


def test_fills_every_medal_to_target(monkeypatch, data_dir, capsys):
    fake = FakeOpenDota()
    monkeypatch.setattr(od, "api_get", fake)
    collect.run(target=5)
    c = load(data_dir)
    usable = c[c["status"] == "usable"].groupby("medal").size()
    assert (usable.reindex(range(1, 9), fill_value=0) >= 5).all()
    assert c["account_id"].is_unique
    assert set(c["status"]) <= {"usable", "too_few", "empty", "pending"}
    assert (c.loc[c["status"] == "too_few", "n_ranked"] == 40).all()
    assert (c.loc[c["status"] == "empty", "n_ranked"] == 0).all()
    # only ranked public matches are opened as seeds
    seeds = {int(p.split("/")[1]) for p, _ in fake.calls if p.startswith("matches/")}
    assert seeds and all((s % 100) % 3 == 0 for s in seeds)  # i % 3 == 0 are the lobby_type 7 ones
    assert "Immortal" in capsys.readouterr().out


def test_player_gets_his_own_medal_not_the_seed_band(monkeypatch, data_dir):
    monkeypatch.setattr(od, "api_get", FakeOpenDota())
    collect.run(target=3)
    c = load(data_dir)
    assert (c["medal"] == c["rank_tier"] // 10).all()
    assert ((c["seed_match_id"] // 1000) != c["medal"]).any()  # some came from another band


def test_resumes_after_daily_limit_without_refetching(monkeypatch, data_dir):
    first = FakeOpenDota(fail_after=25)
    monkeypatch.setattr(od, "api_get", first)
    collect.run(target=5)
    done_first = {p for p, _ in first.calls if p.startswith("players/")}
    assert len(first.calls) == 25

    second = FakeOpenDota()
    monkeypatch.setattr(od, "api_get", second)
    collect.run(target=5)
    done_second = {p for p, _ in second.calls if p.startswith("players/")}
    assert not done_first & done_second  # no player history fetched twice
    seeds_first = [p for p, _ in first.calls if p.startswith("matches/")]
    seeds_second = [p for p, _ in second.calls if p.startswith("matches/")]
    assert not set(seeds_first) & set(seeds_second)
    usable = load(data_dir).query("status == 'usable'").groupby("medal").size()
    assert (usable.reindex(range(1, 9), fill_value=0) >= 5).all()


def test_max_calls_stops_the_run(monkeypatch, data_dir):
    fake = FakeOpenDota()
    monkeypatch.setattr(od, "api_get", fake)
    collect.run(target=50, max_calls=12)
    assert len(fake.calls) == 12


def test_medal_closes_when_pages_run_out(monkeypatch, data_dir):
    monkeypatch.setattr(od, "api_get", FakeOpenDota())
    collect.run(target=10_000)  # unreachable, so every band gets paged to the end
    state = (data_dir / "interim" / "sampling_state.json").read_text()
    assert '"exhausted": [1, 2, 3, 4, 5, 6, 7, 8]' in state


def test_http_error_marks_candidate_and_continues(monkeypatch, data_dir):
    fake = FakeOpenDota()

    def flaky(path, params=None, use_cache=True):
        if path.startswith("players/") and int(path.split("/")[1]) % 10 == 2:
            od.calls_made += 1
            raise od.requests.HTTPError("404")
        return fake(path, params, use_cache)

    monkeypatch.setattr(od, "api_get", flaky)
    collect.run(target=3)
    c = load(data_dir)
    assert (c.loc[c["account_id"] % 10 == 2, "status"].isin(["error", "pending"])).all()
    assert (c["status"] == "error").any()


def test_repeated_page_closes_the_medal(monkeypatch, data_dir):
    def same_page(path, params=None, use_cache=True):
        od.calls_made += 1
        if path == "publicMatches":
            return [{"match_id": 5, "lobby_type": 0}]  # never any ranked seed, never moves
        raise AssertionError(path)

    monkeypatch.setattr(od, "api_get", same_page)
    collect.run(target=1)
    state = (data_dir / "interim" / "sampling_state.json").read_text()
    assert '"exhausted": [1, 2, 3, 4, 5, 6, 7, 8]' in state


def test_stops_cleanly_when_quota_nearly_used(monkeypatch, data_dir):
    fake = FakeOpenDota()

    def counting(path, params=None, use_cache=True):
        out = fake(path, params, use_cache)
        od.last_limits = {"x-rate-limit-remaining-day": str(max(0, 20 - len(fake.calls)))}
        return out

    monkeypatch.setattr(od, "last_limits", {})
    monkeypatch.setattr(od, "api_get", counting)
    collect.run(target=50)
    assert len(fake.calls) == 15  # stops once 5 calls are left


def test_only_window_matches_make_a_player_usable(monkeypatch, data_dir):
    fake = FakeOpenDota()

    def old_histories(path, params=None, use_cache=True):
        out = fake(path, params, use_cache)
        if path.startswith("players/") and int(path.split("/")[1]) % 10 == 4:
            # 150 games, but only 60 after the window start
            out = [{"match_id": k, "start_time": WINDOW_START_TS + (60 - k) * 3600 - 1} for k in range(150)]
        return out

    monkeypatch.setattr(od, "api_get", old_histories)
    collect.run(target=3)
    c = load(data_dir)
    old = c[(c["account_id"] % 10 == 4) & (c["status"] != "pending")]
    assert len(old) and (old["status"] == "too_few").all()
    assert (old["n_ranked"] == 150).all() and (old["n_window"] == 60).all()


def test_rechecks_old_candidates_from_cache_without_api(monkeypatch, data_dir):
    (data_dir / "interim").mkdir(parents=True)
    pd.DataFrame([
        {"account_id": 11, "medal": 1, "rank_tier": 12, "seed_match_id": 1, "status": "usable", "n_ranked": 150},
        {"account_id": 12, "medal": 1, "rank_tier": 12, "seed_match_id": 1, "status": "usable", "n_ranked": 150},
    ]).to_csv(data_dir / "interim" / "candidates.csv", index=False)  # old file, no n_window column
    for acc, t in [(11, RECENT), (12, WINDOW_START_TS - 86400)]:
        f = od.cache_file(f"players/{acc}/matches", HISTORY_PARAMS)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(pd.Series([{"match_id": k, "start_time": t - k * 60} for k in range(150)]).to_json(orient="values"))

    monkeypatch.setattr(od, "api_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no API")))
    collect.run(target=1, max_calls=0)
    c = load(data_dir).set_index("account_id")
    assert c.loc[11, "status"] == "usable" and c.loc[11, "n_window"] == 150
    assert c.loc[12, "status"] == "too_few" and c.loc[12, "n_window"] == 0

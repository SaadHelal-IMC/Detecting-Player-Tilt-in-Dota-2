"""Offline tests for the API client. requests.get and time.sleep are faked,
so these run without network and without waiting.

    python -m pytest -q
"""
import importlib

import pytest

from src.collect import opendota as od


class FakeResponse:
    def __init__(self, status, data=None, headers=None):
        self.status_code = status
        self._data = data
        self.headers = headers or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        raise od.requests.HTTPError(f"{self.status_code}")


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Queue of responses + log of calls, cache pointed at a temp folder."""
    state = {"queue": [], "calls": []}

    def fake_get(url, params=None, timeout=None):
        state["calls"].append((url, dict(params or {})))
        return state["queue"].pop(0)

    monkeypatch.setattr(od.requests, "get", fake_get)
    monkeypatch.setattr(od.time, "sleep", lambda s: None)
    monkeypatch.setattr(od, "RAW_DIR", tmp_path)
    monkeypatch.setattr(od, "API_KEY", None)
    return state


def test_second_call_comes_from_cache(fake):
    fake["queue"].append(FakeResponse(200, [{"match_id": 1}]))
    a = od.api_get("players/1/matches", {"limit": 5})
    b = od.api_get("players/1/matches", {"limit": 5})
    assert a == b == [{"match_id": 1}]
    assert len(fake["calls"]) == 1


def test_different_params_are_different_cache_files(fake):
    fake["queue"] += [FakeResponse(200, [1]), FakeResponse(200, [2])]
    assert od.api_get("players/1/matches", {"limit": 5}) == [1]
    assert od.api_get("players/1/matches", {"limit": 6}) == [2]
    assert len(fake["calls"]) == 2


def test_api_key_is_sent_but_not_part_of_cache_name(fake, monkeypatch):
    monkeypatch.setattr(od, "API_KEY", "secret")
    fake["queue"].append(FakeResponse(200, {"ok": True}))
    od.api_get("players/1", {})
    assert fake["calls"][0][1]["api_key"] == "secret"
    assert "secret" not in str(od.cache_file("players/1", {}))


def test_retries_after_429_then_succeeds(fake):
    fake["queue"] += [
        FakeResponse(429, headers={"X-Rate-Limit-Remaining-Day": "1500"}),
        FakeResponse(200, {"ok": True}),
    ]
    assert od.api_get("players/2") == {"ok": True}
    assert len(fake["calls"]) == 2


def test_stops_when_daily_quota_is_gone(fake):
    fake["queue"].append(FakeResponse(429, headers={"X-Rate-Limit-Remaining-Day": "0"}))
    with pytest.raises(od.DailyLimitReached):
        od.api_get("players/3")


def test_retries_server_errors(fake):
    fake["queue"] += [FakeResponse(502), FakeResponse(200, [])]
    assert od.api_get("publicMatches", {"min_rank": 10}) == []


def test_404_is_not_retried(fake):
    fake["queue"].append(FakeResponse(404))
    with pytest.raises(od.requests.HTTPError):
        od.api_get("players/4")
    assert len(fake["calls"]) == 1


def test_gives_up_after_max_retries(fake):
    fake["queue"] += [FakeResponse(500)] * od.MAX_RETRIES
    with pytest.raises(RuntimeError):
        od.api_get("players/5")


def test_rate_limit_headers_are_kept(fake):
    fake["queue"].append(FakeResponse(200, {}, {"X-Rate-Limit-Remaining-Minute": "59", "Content-Type": "json"}))
    od.api_get("players/6")
    assert od.last_limits == {"x-rate-limit-remaining-minute": "59"}


def test_data_dir_can_be_moved_with_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TILT_DATA_DIR", str(tmp_path))
    importlib.reload(od)
    try:
        assert od.RAW_DIR == tmp_path / "raw"
    finally:
        monkeypatch.delenv("TILT_DATA_DIR")
        importlib.reload(od)
    assert od.RAW_DIR == od.REPO_ROOT / "data" / "raw"

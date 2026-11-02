"""Small OpenDota API client: disk cache, rate limiting and retries.

Every successful response is saved under data/raw/, so a run that crashes or
hits the daily limit can simply be started again and only fetches what is
still missing. Nothing here is clever on purpose, it is one function.

Free tier is roughly 60 calls per minute and 2,000 per day. Set the
environment variable OPENDOTA_API_KEY if you ever buy a key, it is sent with
every request but never written into the cache file names.

Set TILT_DATA_DIR to keep the data folder outside the repo, e.g. outside
OneDrive, so thousands of cache files do not get synced.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://api.opendota.com/api"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("TILT_DATA_DIR") or REPO_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
API_KEY = os.environ.get("OPENDOTA_API_KEY")

MIN_INTERVAL = 1.1   # seconds between two calls, keeps us under 60/min
MAX_RETRIES = 5
TIMEOUT = 30

# simple module state, good enough for one script at a time
last_call_time = 0.0
calls_made = 0
last_limits = {}     # rate-limit headers of the latest response


class DailyLimitReached(Exception):
    """Raised when OpenDota says the daily quota is used up. Rerun tomorrow."""


def cache_file(path, params):
    path = path.strip("/")
    key = path + "?" + json.dumps(params, sort_keys=True)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    folder = RAW_DIR / path.split("/")[0]
    return folder / f"{path.replace('/', '_')}__{digest}.json"


def api_get(path, params=None, use_cache=True):
    """GET an OpenDota endpoint, e.g. api_get("players/123/matches", {"limit": 10}).

    Lists in params (like project=[...]) are sent as repeated keys, which is
    what OpenDota expects. Returns the parsed JSON.
    """
    global last_call_time, calls_made, last_limits

    params = dict(params or {})
    f = cache_file(path, params)
    if use_cache and f.exists():
        return json.loads(f.read_text(encoding="utf-8"))

    send = dict(params)
    if API_KEY:
        send["api_key"] = API_KEY
    url = f"{BASE_URL}/{path.strip('/')}"

    for attempt in range(MAX_RETRIES):
        wait = MIN_INTERVAL - (time.monotonic() - last_call_time)
        if wait > 0:
            time.sleep(wait)
        last_call_time = time.monotonic()

        try:
            r = requests.get(url, params=send, timeout=TIMEOUT)
        except requests.RequestException as e:
            pause = 5 * 2 ** attempt
            print(f"  network error on {path} ({e}), retry in {pause}s")
            time.sleep(pause)
            continue

        calls_made += 1
        last_limits = {k.lower(): v for k, v in r.headers.items() if "rate-limit" in k.lower()}

        if r.status_code == 200:
            data = r.json()
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(data), encoding="utf-8")
            return data

        if r.status_code == 429:
            if last_limits.get("x-rate-limit-remaining-day") == "0":
                raise DailyLimitReached(f"daily quota used up after {calls_made} calls this run")
            print(f"  429 on {path}, waiting 60s")
            time.sleep(60)
            continue

        if r.status_code >= 500:
            pause = 5 * 2 ** attempt
            print(f"  {r.status_code} on {path}, retry in {pause}s")
            time.sleep(pause)
            continue

        # 4xx other than 429 will not get better by retrying
        r.raise_for_status()

    raise RuntimeError(f"giving up on {path} after {MAX_RETRIES} attempts")

"""Sample players per medal and download their ranked match history.

Per medal (Herald .. Immortal) until each has TARGET usable players:
  1. page back through /publicMatches for the medal's rank band and keep the
     ranked matches (lobby_type 7) as seeds; most public matches are Turbo or
     unranked, so only about 10 of 100 per page are seeds
  2. /matches/{seed} lists the 10 players; every public one with a visible
     rank_tier becomes a candidate for THEIR OWN medal (ranks inside one match
     can differ a lot, and this fills other medals for free)
  3. /players/{id}/matches downloads the candidate's latest 500 ranked games;
     with at least --min-ranked of them inside the analysis window (from
     WINDOW_START, see constants.py) the player is "usable"
The medal with the fewest usable players is always served first.

Raw responses are cached by the API client and progress is saved after every
call to data/interim/candidates.csv and sampling_state.json. Just run it
again (e.g. next day after the quota resets) and it continues where it
stopped. Don't keep candidates.csv open in Excel while it runs, Windows then
blocks the file and the save fails.

    python -m src.collect.collect                   # 150 usable players per medal
    python -m src.collect.collect --target 20       # dry run
    python -m src.collect.collect --max-calls 300   # stop after 300 API calls
"""
import argparse
import json
import math
import os

import pandas as pd

from src.collect import opendota as od
from src.collect.constants import HISTORY_PARAMS, MEDALS, WINDOW_START_TS

CAND_COLS = ["account_id", "medal", "rank_tier", "seed_match_id", "status", "n_ranked", "n_window"]


def in_window(hist):
    """Number of matches in a history that fall inside the analysis window."""
    return sum(1 for r in hist or [] if (r.get("start_time") or 0) >= WINDOW_START_TS)


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run(target=150, min_ranked=100, max_calls=None):
    folder = od.DATA_DIR / "interim"
    folder.mkdir(parents=True, exist_ok=True)
    cand_file = folder / "candidates.csv"
    state_file = folder / "sampling_state.json"

    # --- load progress -----------------------------------------------------
    cands = pd.read_csv(cand_file).to_dict("records") if cand_file.exists() else []
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    cursor = {int(k): v for k, v in state.get("cursor", {}).items()}         # less_than_match_id per medal
    seed_queue = {int(k): v for k, v in state.get("seed_queue", {}).items()}  # ranked seeds not used yet
    seeds_done = set(state.get("seeds_done", []))
    exhausted = set(state.get("exhausted", []))                               # medals with no more pages
    known = {int(c["account_id"]) for c in cands}

    # candidates fetched before the window rule existed: recount from the
    # local cache (no API calls) so the quotas only count window matches
    rechecked = 0
    for c in cands:
        if c["status"] in ("usable", "too_few") and pd.isna(c.get("n_window")):
            f = od.cache_file(f"players/{int(c['account_id'])}/matches", HISTORY_PARAMS)
            if f.exists():
                c["n_window"] = in_window(json.loads(f.read_text(encoding="utf-8")))
                c["status"] = "usable" if c["n_window"] >= min_ranked else "too_few"
                rechecked += 1
    if rechecked:
        print(f"re-checked {rechecked} downloaded players against the window from cache")
    calls_at_start = od.calls_made

    def save():
        write_atomic(cand_file, pd.DataFrame(cands, columns=CAND_COLS).to_csv(index=False))
        write_atomic(state_file, json.dumps({
            "cursor": cursor, "seed_queue": seed_queue,
            "seeds_done": sorted(seeds_done), "exhausted": sorted(exhausted),
        }))

    def count(medal, status):
        return sum(1 for c in cands if c["medal"] == medal and c["status"] == status)

    def log(msg):
        print(f"[{od.calls_made - calls_at_start:>4} calls] {msg}")

    try:
        while True:
            open_medals = [m for m in MEDALS if count(m, "usable") < target and m not in exhausted]
            if not open_medals:
                break
            if max_calls is not None and od.calls_made - calls_at_start >= max_calls:
                log(f"stopping, --max-calls {max_calls} reached")
                break
            left_today = od.last_limits.get("x-rate-limit-remaining-day")
            if left_today is not None and int(left_today) <= 5:
                log(f"stopping, only {left_today} calls left today. Run again after the quota resets.")
                break
            m = min(open_medals, key=lambda x: count(x, "usable"))  # thinnest medal first

            pending = [c for c in cands if c["medal"] == m and c["status"] == "pending"]
            if pending:
                # step 3: download one candidate's history
                c = pending[0]
                try:
                    hist = od.api_get(f"players/{int(c['account_id'])}/matches", HISTORY_PARAMS)
                except od.requests.HTTPError as e:
                    c["status"] = "error"
                    log(f"{MEDALS[m]:<8} {int(c['account_id']):>10}: {e} -> error")
                    save()
                    continue
                n, nw = len(hist or []), in_window(hist)
                c["n_ranked"], c["n_window"] = n, nw
                c["status"] = "usable" if nw >= min_ranked else ("empty" if n == 0 else "too_few")
                log(f"{MEDALS[m]:<8} {int(c['account_id']):>10}: {n:>3} ranked, {nw:>3} in window"
                    f" -> {c['status']:<7} ({count(m, 'usable')}/{target})")

            elif seed_queue.get(m):
                # step 2: open one seed match, collect its public players
                seed = seed_queue[m][0]  # popped only after the call worked
                try:
                    match = od.api_get(f"matches/{seed}") or {}
                except od.requests.HTTPError as e:
                    match = {}
                    log(f"{MEDALS[m]:<8} seed {seed}: {e}, skipped")
                seed_queue[m].pop(0)
                seeds_done.add(seed)
                added = 0
                for p in match.get("players", []):
                    acc, rt = p.get("account_id"), p.get("rank_tier")
                    if acc is None or rt is None or (isinstance(rt, float) and math.isnan(rt)):
                        continue
                    acc, med = int(acc), int(rt) // 10
                    if med not in MEDALS or acc in known:
                        continue
                    known.add(acc)
                    cands.append({"account_id": acc, "medal": med, "rank_tier": int(rt),
                                  "seed_match_id": seed, "status": "pending",
                                  "n_ranked": None, "n_window": None})
                    added += 1
                log(f"{MEDALS[m]:<8} seed {seed}: +{added} candidates")

            else:
                # step 1: next page of public matches in this medal's band
                params = {"min_rank": m * 10, "max_rank": m * 10 + 5}
                if cursor.get(m):
                    params["less_than_match_id"] = cursor[m]
                page = od.api_get("publicMatches", params) or []
                if not page:
                    exhausted.add(m)
                    log(f"{MEDALS[m]:<8} no more public matches, medal closed")
                    continue
                new_cursor = min(r["match_id"] for r in page)
                if cursor.get(m) and new_cursor >= cursor[m]:
                    exhausted.add(m)  # API repeated a page, paging is not moving back
                    log(f"{MEDALS[m]:<8} paging stopped moving, medal closed")
                    continue
                cursor[m] = new_cursor
                seed_queue[m] = [r["match_id"] for r in page
                                 if r.get("lobby_type") == 7 and r["match_id"] not in seeds_done]
                log(f"{MEDALS[m]:<8} publicMatches page: {len(seed_queue[m])} ranked seeds of {len(page)}")
            save()

    except od.DailyLimitReached as e:
        print(f"\n{e}. Run the same command again after the quota resets.")
    finally:
        save()

    # --- summary -------------------------------------------------------------
    rows = [{"medal": MEDALS[m], "usable": count(m, "usable"), "too_few": count(m, "too_few"),
             "empty": count(m, "empty"), "pending": count(m, "pending")} for m in MEDALS]
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print(f"\nAPI calls this run: {od.calls_made - calls_at_start}, "
          f"left today: {od.last_limits.get('x-rate-limit-remaining-day', '?')}")
    return cands


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=150, help="usable players per medal")
    ap.add_argument("--min-ranked", type=int, default=100, help="ranked matches needed to be usable")
    ap.add_argument("--max-calls", type=int, default=None, help="stop after this many API calls")
    a = ap.parse_args()
    run(a.target, a.min_ranked, a.max_calls)

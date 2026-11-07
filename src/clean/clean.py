"""Turn the cached raw histories into one tidy match table.

Reads only the local cache and never calls the API, so it can run while the
collector is still going and simply uses what is downloaded so far.

Steps
  1. usable players from data/interim/candidates.csv
  2. their cached /players/{id}/matches answer -> one row per (player, match)
  3. win, side, start datetime, short-game flag (remakes under 10 minutes)
  4. drop duplicates, anything not ranked and anything before the analysis
     window (WINDOW_START in constants.py), sort each player by time
  5. split each history into runs at breaks longer than 30 days and drop runs
     shorter than 20 matches (a few isolated old games say nothing about the
     player's current baseline)
  6. keep players with at least 100 matches left
  7. bracket = the player's profile medal at sampling time (lobby
     average_rank caps at about 75 and cannot show Immortal, see
     DECISIONS.md); the median lobby medal of the last 100 matches is kept
     as lobby_medal for robustness checks

Output
  data/processed/matches.parquet        one row per (account_id, match_id)
  data/processed/players.parquet        one row per player
  results/tables/dataset_summary.csv    players and matches per bracket

    python -m src.clean.clean
"""
import json

import pandas as pd

from src.collect import opendota as od
from src.collect.constants import HISTORY_PARAMS, MATCH_FIELDS, MEDALS, WINDOW_START, WINDOW_START_TS

MIN_MATCHES = 100    # per player after cleaning
MIN_RUN = 20         # matches in a run between long breaks
BREAK_DAYS = 30      # a pause longer than this starts a new run
SHORT_GAME_S = 600   # remakes / early abandons


def run():
    # --- 1-2: usable players and their cached histories ----------------------
    cands = pd.read_csv(od.DATA_DIR / "interim" / "candidates.csv")
    usable = cands[cands["status"] == "usable"]
    frames, missing = [], 0
    for acc, medal in zip(usable["account_id"], usable["medal"]):
        f = od.cache_file(f"players/{int(acc)}/matches", HISTORY_PARAMS)
        if not f.exists():
            missing += 1
            continue
        d = pd.DataFrame(json.loads(f.read_text(encoding="utf-8")))
        if d.empty:
            continue
        d.insert(0, "account_id", int(acc))
        d["profile_medal"] = int(medal)
        frames.append(d.dropna(axis=1, how="all"))  # all-null columns come back via concat
    if not frames:
        raise SystemExit("no cached histories found, run the collector first")

    df = pd.concat(frames, ignore_index=True)
    df = df.reindex(columns=["account_id"] + MATCH_FIELDS + ["profile_medal"])  # missing fields -> NaN
    for c in df.columns:  # JSON nulls can leave numeric columns as object dtype
        if c != "radiant_win":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    log = [("raw rows", len(df))]

    # --- 3-4: basic columns, duplicates, ranked only, time order -------------
    df = df.drop_duplicates(["account_id", "match_id"])
    log.append(("after duplicates", len(df)))
    df = df[df["lobby_type"] == 7]
    log.append(("after ranked only", len(df)))
    df = df[df["start_time"] >= WINDOW_START_TS]
    log.append((f"after window from {WINDOW_START}", len(df)))
    df = df.sort_values(["account_id", "start_time", "match_id"]).reset_index(drop=True)

    df["is_radiant"] = df["player_slot"] < 128
    df["win"] = df["is_radiant"] == df["radiant_win"]
    df["start"] = pd.to_datetime(df["start_time"], unit="s", utc=True)
    df["short_game"] = df["duration"] < SHORT_GAME_S
    df["parsed"] = df["version"].notna()
    df["match_medal"] = (df["average_rank"] // 10).clip(1, 8)

    # --- 5: runs between long breaks -----------------------------------------
    gap_days = df.groupby("account_id")["start_time"].diff() / 86400
    df["run_id"] = (gap_days > BREAK_DAYS).astype(int).groupby(df["account_id"]).cumsum()
    run_len = df.groupby(["account_id", "run_id"])["match_id"].transform("size")
    df = df[run_len >= MIN_RUN]
    log.append((f"after runs < {MIN_RUN} dropped", len(df)))

    # --- 6: enough matches per player ----------------------------------------
    n = df.groupby("account_id")["match_id"].transform("size")
    df = df[n >= MIN_MATCHES].copy()
    log.append((f"after players < {MIN_MATCHES} dropped", len(df)))
    df["match_idx"] = df.groupby("account_id").cumcount()  # 0 = oldest match

    # --- 7: player table and bracket -----------------------------------------
    df["solo"] = (df["party_size"] == 1).astype(float).where(df["party_size"].notna())
    players = df.groupby("account_id").agg(
        n_matches=("match_id", "size"),
        first_match=("start", "min"),
        last_match=("start", "max"),
        median_avg_rank=("average_rank", "median"),
        profile_medal=("profile_medal", "first"),
        win_rate=("win", "mean"),
        solo_share=("solo", "mean"),
        parsed_share=("parsed", "mean"),
        short_games=("short_game", "sum"),
        n_runs=("run_id", "nunique"),
    ).reset_index()
    recent = df[df.groupby("account_id")["match_idx"].transform("max") - df["match_idx"] < 100]
    lobby = (recent.groupby("account_id")["average_rank"].median() // 10).clip(1, 8)
    players["lobby_medal"] = players["account_id"].map(lobby)
    players["bracket"] = players["profile_medal"].astype(int)
    players["bracket_name"] = players["bracket"].map(MEDALS)
    players["span_days"] = (players["last_match"] - players["first_match"]).dt.days
    df = df.merge(players[["account_id", "bracket"]], on="account_id", how="left")

    # --- save ------------------------------------------------------------------
    out = od.DATA_DIR / "processed"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "matches.parquet", index=False)
    players.to_parquet(out / "players.parquet", index=False)

    summary = players.groupby("bracket").agg(
        players=("account_id", "size"),
        matches=("n_matches", "sum"),
        median_matches=("n_matches", "median"),
        median_span_days=("span_days", "median"),
        mean_win_rate=("win_rate", "mean"),
        mean_solo_share=("solo_share", "mean"),
        lobby_within_1=("lobby_medal", lambda s: ((s - players.loc[s.index, "bracket"]).abs() <= 1).mean()),
    ).reset_index()
    summary.insert(1, "bracket_name", summary["bracket"].map(MEDALS))
    tables = od.REPO_ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary.round(3).to_csv(tables / "dataset_summary.csv", index=False)

    # --- report ----------------------------------------------------------------
    print(f"usable players in candidates.csv: {len(usable)}, history not cached yet: {missing}")
    for step, rows in log:
        print(f"  {step:<28} {rows:>8,} rows")
    print(f"players kept: {len(players)}, matches kept: {len(df):,}")
    nulls = df[MATCH_FIELDS].isna().mean()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print("\nshare of nulls (fields with any):")
        print(nulls.map(lambda x: f"{x:.1%}").to_string())
    print(f"\nshort games (< {SHORT_GAME_S // 60} min): {df['short_game'].sum():,}")
    print("\n" + summary.round(2).to_string(index=False))
    print(f"\nsaved {out / 'matches.parquet'}\n      {out / 'players.parquet'}\n      {tables / 'dataset_summary.csv'}")
    return df, players


if __name__ == "__main__":
    run()

"""Day-1 check: pull one account's ranked history and see what OpenDota gives us.

Answers three questions before the real collector is written:
  1. which of the fields we want actually come back (and how often they are null)
  2. what the rate-limit headers look like
  3. whether the numbers look sane (dates, win rate, gaps between games)

Usage (from the repo root):
    python -m src.collect.check_fields 123456789

The number is the Steam32 account id, the one in your OpenDota profile URL
(opendota.com/players/<id>). The profile needs "Expose Public Match Data" on.
"""
import sys

import pandas as pd

from src.collect import opendota as od

MEDALS = {1: "Herald", 2: "Guardian", 3: "Crusader", 4: "Archon",
          5: "Legend", 6: "Ancient", 7: "Divine", 8: "Immortal"}

# everything we might use; when `project` is passed OpenDota returns only these
FIELDS = [
    "match_id", "player_slot", "radiant_win", "start_time", "duration",
    "game_mode", "lobby_type", "hero_id", "kills", "deaths", "assists",
    "gold_per_min", "xp_per_min", "last_hits", "denies", "hero_damage",
    "tower_damage", "hero_healing", "leaver_status", "party_size",
    "average_rank", "version",
]

if len(sys.argv) != 2:
    sys.exit("usage: python -m src.collect.check_fields <steam32_account_id>")
account_id = int(sys.argv[1])

# --- profile ---------------------------------------------------------------
profile = od.api_get(f"players/{account_id}", use_cache=False)
name = (profile.get("profile") or {}).get("personaname")
rank_tier = profile.get("rank_tier")
medal = MEDALS.get(rank_tier // 10, "?") if rank_tier else "unranked"
print(f"account {account_id}: {name}, rank_tier={rank_tier} ({medal})")
print("rate-limit headers:", od.last_limits or "none sent")

# --- ranked match history --------------------------------------------------
matches = od.api_get(
    f"players/{account_id}/matches",
    {"lobby_type": 7, "limit": 500, "project": FIELDS},
    use_cache=False,
)
print("rate-limit headers:", od.last_limits or "none sent")

if not matches:
    sys.exit("no ranked matches came back: profile private, wrong id, or no ranked games")

df = pd.DataFrame(matches)
print(f"\n{len(df)} ranked matches returned")

# --- field coverage --------------------------------------------------------
print("\nfield coverage (share of matches where the field is not null):")
missing = [c for c in FIELDS if c not in df.columns]
for c in FIELDS:
    share = df[c].notna().mean() if c in df.columns else 0.0
    print(f"  {c:<15} {share:6.1%}")
if missing:
    print("never returned:", ", ".join(missing))
extra = [c for c in df.columns if c not in FIELDS]
if extra:
    print("returned but not asked for:", ", ".join(extra))

# --- sanity numbers ----------------------------------------------------------
df = df.sort_values("start_time").reset_index(drop=True)
df["start"] = pd.to_datetime(df["start_time"], unit="s", utc=True)
df["win"] = (df["player_slot"] < 128) == df["radiant_win"]
gap_min = (df["start_time"] - df["start_time"].shift(1) - df["duration"].shift(1)) / 60

print(f"\nfirst match  {df['start'].iloc[0]:%Y-%m-%d}")
print(f"last match   {df['start'].iloc[-1]:%Y-%m-%d}")
print(f"win rate     {df['win'].mean():.1%}")
print(f"heroes used  {df['hero_id'].nunique()}")
print(f"gap between games, minutes: median {gap_min.median():.1f}, "
      f"25% {gap_min.quantile(0.25):.1f}, share under 2 h {(gap_min < 120).mean():.1%}")
if "version" in df.columns:
    print(f"parsed share {df['version'].notna().mean():.1%}")

out = od.DATA_DIR / "interim" / "own_matches_check.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"\nsaved {out}")
print(f"API calls used: {od.calls_made}")

"""Setup probe for player sampling: what do /publicMatches and /matches/{id} return?

Spends 2 API calls. Tells us how the sampler (Wed) has to work:
  - does /publicMatches filter by rank and give an average rank per match
  - how many of the 10 players in a match are public (have an account_id)
  - whether rank_tier comes with the match or needs an extra /players call

Usage (from the repo root):
    python -m src.collect.probe_sampling          # Legend band by default
    python -m src.collect.probe_sampling 80 85    # Immortal band
"""
import sys

import pandas as pd

from src.collect import opendota as od

min_rank = int(sys.argv[1]) if len(sys.argv) > 1 else 50
max_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 59

# --- recent public matches in one rank band ----------------------------------
pm = od.api_get("publicMatches", {"min_rank": min_rank, "max_rank": max_rank}, use_cache=False)
print("rate-limit headers:", od.last_limits or "none sent")
if not pm:
    sys.exit("publicMatches returned nothing")

pm = pd.DataFrame(pm)
print(f"\n/publicMatches min_rank={min_rank} max_rank={max_rank}: {len(pm)} matches")
print("columns:", ", ".join(pm.columns))
for c in ["avg_rank_tier", "num_rank_tier", "lobby_type", "game_mode"]:
    if c in pm.columns:
        print(f"  {c:<14} values: {pm[c].value_counts().head(8).to_dict()}")
if "start_time" in pm.columns:
    age_h = (pd.Timestamp.now(tz="UTC").timestamp() - pm["start_time"]) / 3600
    print(f"  match age in hours: newest {age_h.min():.1f}, oldest {age_h.max():.1f}")

# --- one full match: who in it is public? ------------------------------------
match_id = int(pm["match_id"].iloc[0])
m = od.api_get(f"matches/{match_id}", use_cache=False)
players = pd.DataFrame(m.get("players", []))
print(f"\n/matches/{match_id}: {len(players)} players")
keep = [c for c in ["account_id", "rank_tier", "personaname", "player_slot", "hero_id"] if c in players.columns]
print(players[keep].to_string(index=False))
n_public = players["account_id"].notna().sum() if "account_id" in players.columns else 0
n_rank = players["rank_tier"].notna().sum() if "rank_tier" in players.columns else 0
print(f"\npublic account_id: {n_public}/10, rank_tier shown: {n_rank}/10")
print(f"API calls used: {od.calls_made}")

"""Shared constants for the collection scripts."""

MEDALS = {1: "Herald", 2: "Guardian", 3: "Crusader", 4: "Archon",
          5: "Legend", 6: "Ancient", 7: "Divine", 8: "Immortal"}

# per-match fields asked from /players/{id}/matches. When `project` is passed
# OpenDota returns only these, so the defaults have to be listed too.
# All of them come back (see DECISIONS.md, section 2).
MATCH_FIELDS = [
    "match_id", "player_slot", "radiant_win", "start_time", "duration",
    "game_mode", "lobby_type", "hero_id", "kills", "deaths", "assists",
    "gold_per_min", "xp_per_min", "last_hits", "denies", "hero_damage",
    "tower_damage", "hero_healing", "leaver_status", "party_size",
    "average_rank", "version",
]

# one call = a player's latest 500 ranked matches with all fields above
HISTORY_PARAMS = {"lobby_type": 7, "limit": 500, "project": MATCH_FIELDS}

# analysis window: only ranked matches from this date on count (two years
# before collection; older histories reach back to 2015, span many patches
# and have no average_rank before 2022, see DECISIONS.md section 4)
WINDOW_START = "2024-11-01"
WINDOW_START_TS = 1730419200  # 2024-11-01 00:00 UTC

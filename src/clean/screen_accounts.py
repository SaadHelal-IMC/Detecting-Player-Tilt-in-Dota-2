"""Screen the sampled accounts for bots, shared accounts and sold accounts.

Bot farms reached ranked Dota in 2026, and a bought or shared account mixes
two people's play in one history. Neither is tilt. So every player gets a few
simple account-level signals from matches.parquet (no API calls), and a player
with at least MIN_SIGNALS of them is marked as a suspect. Suspects are NOT
dropped: the main results keep them and the sensitivity step reruns SQ2
without them.

The screen looks at the whole window at once. That is fine here because it
only describes the sample, it is never a detector input.

Signals (one row per player)
  quiet_share   share of matches started in the quietest 6-hour block of the
                day (UTC), per 30-day block with at least 30 matches, median
                over those blocks. People sleep, a 24/7 account does not
                (evenly spread play gives 0.25). Per block, because over the
                whole window a player whose schedule moved (new job,
                semester) looks like someone who plays around the clock
  max_24h       most matches started within any 24 hours
  spread        how much GPM, duration and KDA vary: geometric mean of their
                robust CVs (1.4826 * MAD / median). Scripted play is unusually
                uniform, so this one is flagged when LOW
  top3_share    share of matches on the player's 3 most played heroes
  party         stack5_share OR shared_share above its threshold, counted as
                ONE signal (both measure playing in a fixed group, and they
                mostly fire together)
    stack5_share  share of 5-stacks among matches with known party_size
    shared_share  share of matches that contain another sampled player. Seed
                  matches are left out, players found in the same seed match
                  share that one by design
  shift         strongest change at one point of the history in heroes AND
                play hours: max over split points of min(hero TVD, hour TVD)
                between all matches before and all matches after (sold
                account). shift_at is the date of that split
  boost_wr      best win rate over BOOST_WINDOW matches in a row (smurf or
                boosted account; matchmaking keeps honest players near 50%).
                boost_end is the date of the last match of that stretch
  throw_days    UTC days with at least THROW_MIN_GAMES games, at most
                THROW_MAX_WR wins and a median KDA below THROW_MAX_KDA
                (losing on purpose, e.g. to derank). first_throw_day is the
                first of them

Thresholds sit at about the 99th percentile of the 1,180-player sample (spread
at a level only scripted play reaches), see DECISIONS.md section 10.
boost_wr and throw_days are direct evidence, so either one alone marks a
suspect; the other signals need MIN_SIGNALS.

cv_group joins players who share at least LINK_MATCHES matches (seeds
included), so CV can keep them in one fold: their shared matches have the
same outcome and timing. Everyone else is a group of one.

Output
  data/processed/account_screen.parquet       all signals per player
  data/processed/players.parquet              + n_signals, suspect_account, cv_group
  data/interim/account_screen_check.csv       flagged players with OpenDota
                                              and Stratz links for the hand check
  results/tables/account_screen_summary.csv   counts per bracket only

Account ids next to a "suspect" label stay in the data folder, outside the
public repo. Only per-bracket counts go into results/.

After the hand check, write data/interim/account_screen_review.csv with the
columns account_id, verdict (suspect or clear) and note. A verdict there
overrides the automatic flag on the next run.

Run it after clean.py (clean rewrites players.parquet without the columns):

    python -m src.clean.screen_accounts
"""
import numpy as np
import pandas as pd

from src.collect import opendota as od
from src.collect.constants import MEDALS

MIN_SIGNALS = 2       # signals needed to mark a suspect
SHIFT_SIDE = 50       # matches needed on each side of a split point
SHIFT_STEP = 5        # split points every 5 matches
MIN_PARTY_KNOWN = 20  # matches with known party_size needed for stack5_share
BLOCK_DAYS = 30       # quiet_share is taken per block of this many days
MIN_BLOCK = 30        # matches a block needs to count for quiet_share
LINK_MATCHES = 10     # shared matches that put two players in one cv_group
BOOST_WINDOW = 100    # matches in a row for boost_wr
THROW_MIN_GAMES = 8   # a throw day has at least this many games,
THROW_MAX_WR = 0.2    # at most this share of wins
THROW_MAX_KDA = 0.5   # and a median KDA below this

# about the 99th percentile of the sample, see DECISIONS.md section 10
HIGH = {"quiet_share": 0.15, "max_24h": 27, "top3_share": 0.95,
        "stack5_share": 0.50, "shared_share": 0.50, "shift": 0.60,
        "boost_wr": 0.75, "throw_days": 0}
LOW = {"spread": 0.12}
SIGNALS = ["quiet_share", "max_24h", "spread", "top3_share",
           "stack5_share", "shared_share", "shift", "boost_wr", "throw_days"]
FLAGS = ["quiet_share", "max_24h", "spread", "top3_share", "party", "shift",
         "boost_wr", "throw_days"]
STRONG = ["boost_wr", "throw_days"]  # each one alone marks a suspect


def rcv(x):
    """Robust coefficient of variation, 1.4826 * MAD / median."""
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) / med if med > 0 else np.nan


def quiet(hours):
    """Share of matches in the quietest 6-hour block of the day."""
    per_hour = np.bincount(hours, minlength=24)
    six_h = np.convolve(np.r_[per_hour, per_hour[:5]], np.ones(6), mode="valid")  # 24 blocks, wraps midnight
    return six_h.min() / len(hours)


def tvd_at_splits(codes, n_bins, ks):
    """Total variation distance between the matches before and after each split k."""
    n = len(codes)
    onehot = np.zeros((n, n_bins))
    onehot[np.arange(n), codes] = 1
    before = onehot.cumsum(axis=0)[ks - 1]          # counts in matches 0 .. k-1
    after = onehot.sum(axis=0) - before
    return 0.5 * np.abs(before / ks[:, None] - after / (n - ks)[:, None]).sum(axis=1)


def signals(df, seed_ids):
    """One row of screen signals per player from matches.parquet rows."""
    df = df.sort_values(["account_id", "start_time"])
    others = df.groupby("match_id")["account_id"].transform("size") - 1
    df = df.assign(shared=(others > 0) & ~df["match_id"].isin(seed_ids))

    rows = []
    for acc, g in df.groupby("account_id"):
        t = g["start_time"].to_numpy().astype("int64")
        n = len(t)
        hour = (t // 3600) % 24

        block = t // (BLOCK_DAYS * 86400)
        per_block = [quiet(hour[block == b]) for b in np.unique(block) if (block == b).sum() >= MIN_BLOCK]
        in_24h = np.searchsorted(t, t + 86400) - np.arange(n)

        kda = ((g["kills"] + g["assists"]) / g["deaths"].clip(lower=1)).to_numpy()
        cvs = np.array([rcv(g["gold_per_min"].to_numpy()), rcv(g["duration"].to_numpy()), rcv(kda)])
        party = g["party_size"].dropna()
        win = g["win"].to_numpy().astype(float)

        boost_wr, boost_end = np.nan, pd.NaT
        if n >= BOOST_WINDOW:
            roll = np.convolve(win, np.ones(BOOST_WINDOW), mode="valid") / BOOST_WINDOW
            i = roll.argmax()
            boost_wr, boost_end = roll[i], pd.to_datetime(t[i + BOOST_WINDOW - 1], unit="s", utc=True)

        days = pd.DataFrame({"day": t // 86400, "win": win, "kda": kda}).groupby("day").agg(
            games=("win", "size"), wr=("win", "mean"), kda=("kda", "median"))
        throw = days[(days["games"] >= THROW_MIN_GAMES) & (days["wr"] <= THROW_MAX_WR)
                     & (days["kda"] < THROW_MAX_KDA)]

        shift, shift_at = np.nan, pd.NaT
        ks = np.arange(SHIFT_SIDE, n - SHIFT_SIDE + 1, SHIFT_STEP)
        if len(ks):
            heroes = pd.factorize(g["hero_id"])[0]
            both = np.minimum(tvd_at_splits(heroes, heroes.max() + 1, ks),
                              tvd_at_splits(hour // 3, 8, ks))  # 3-hour bins
            i = both.argmax()
            shift, shift_at = both[i], pd.to_datetime(t[ks[i]], unit="s", utc=True)

        rows.append({
            "account_id": acc, "n": n,
            "quiet_share": np.median(per_block) if per_block else np.nan,
            "quiet_blocks": len(per_block),
            "max_24h": in_24h.max(),
            "cv_gpm": cvs[0], "cv_duration": cvs[1], "cv_kda": cvs[2],
            "spread": np.exp(np.log(cvs).mean()) if (cvs > 0).all() else 0.0,
            "top3_share": g["hero_id"].value_counts().head(3).sum() / n,
            "stack5_share": (party == 5).mean() if len(party) >= MIN_PARTY_KNOWN else np.nan,
            "shared_share": g["shared"].mean(),
            "shift": shift, "shift_at": shift_at,
            "boost_wr": boost_wr, "boost_end": boost_end,
            "throw_days": len(throw),
            "first_throw_day": pd.to_datetime(throw.index.min() * 86400, unit="s", utc=True) if len(throw) else pd.NaT,
        })
    s = pd.DataFrame(rows)

    # for the hand check: the other sampled player with the most shared matches
    sh = df.loc[df["shared"], ["account_id", "match_id"]]
    pairs = sh.merge(sh, on="match_id", suffixes=("", "_other"))
    pairs = pairs[pairs["account_id"] != pairs["account_id_other"]]
    top = (pairs.groupby(["account_id", "account_id_other"]).size().rename("shared_max").reset_index()
           .sort_values("shared_max").drop_duplicates("account_id", keep="last")
           .rename(columns={"account_id_other": "shared_partner"}))
    s = s.merge(top, on="account_id", how="left")
    s["shared_max"] = s["shared_max"].fillna(0).astype(int)
    return s


def cv_groups(df):
    """Group id per player: players sharing >= LINK_MATCHES matches end up together."""
    m = df[["account_id", "match_id"]]
    pairs = m.merge(m, on="match_id", suffixes=("", "_other"))
    pairs = pairs[pairs["account_id"] < pairs["account_id_other"]]
    links = pairs.groupby(["account_id", "account_id_other"]).size()
    links = links[links >= LINK_MATCHES].index

    parent = {a: a for a in df["account_id"].unique()}  # union-find
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in links:
        parent[root(a)] = root(b)
    groups = pd.Series({a: root(a) for a in parent})
    return groups.rank(method="dense").astype(int).rename("cv_group")  # 1 .. number of groups


def run():
    proc, interim = od.DATA_DIR / "processed", od.DATA_DIR / "interim"
    df = pd.read_parquet(proc / "matches.parquet")
    players = pd.read_parquet(proc / "players.parquet")
    seeds = set(pd.read_csv(interim / "candidates.csv")["seed_match_id"].dropna().astype("int64"))

    s = signals(df, seeds).merge(players[["account_id", "bracket"]], on="account_id")
    for k, thr in HIGH.items():
        s["flag_" + k] = s[k] > thr                  # NaN never flags
    for k, thr in LOW.items():
        s["flag_" + k] = s[k] < thr
    s["flag_party"] = s["flag_stack5_share"] | s["flag_shared_share"]
    s["n_signals"] = s[["flag_" + k for k in FLAGS]].sum(axis=1)
    s["auto_suspect"] = (s["n_signals"] >= MIN_SIGNALS) | s[["flag_" + k for k in STRONG]].any(axis=1)
    s["cv_group"] = s["account_id"].map(cv_groups(df))

    # hand-check verdicts win over the automatic flag
    s["verdict"] = None
    review = interim / "account_screen_review.csv"
    if review.exists():
        r = pd.read_csv(review)
        r["verdict"] = r["verdict"].str.strip().str.lower()
        bad = set(r["verdict"]) - {"suspect", "clear"}
        if bad:
            raise SystemExit(f"{review.name}: verdict must be suspect or clear, found {bad}")
        s["verdict"] = s["account_id"].map(r.set_index("account_id")["verdict"])
    s["suspect_account"] = s["verdict"].eq("suspect") | (s["auto_suspect"] & s["verdict"].isna())

    # --- save -------------------------------------------------------------------
    new_cols = ["n_signals", "suspect_account", "cv_group"]
    s.to_parquet(proc / "account_screen.parquet", index=False)
    players = players.drop(columns=new_cols, errors="ignore")
    players = players.merge(s[["account_id"] + new_cols], on="account_id", how="left")
    players.to_parquet(proc / "players.parquet", index=False)

    check = s[s["auto_suspect"] | s["suspect_account"]].sort_values("n_signals", ascending=False, kind="stable").copy()
    check.insert(1, "bracket_name", check["bracket"].map(MEDALS))
    check["opendota"] = "https://www.opendota.com/players/" + check["account_id"].astype(str)
    check["stratz"] = "https://stratz.com/players/" + check["account_id"].astype(str)
    num = check.select_dtypes("number").columns
    check[num] = check[num].round(3)
    check.to_csv(interim / "account_screen_check.csv", index=False)

    summary = s.groupby("bracket").agg(players=("account_id", "size"),
                                       **{k: ("flag_" + k, "sum") for k in FLAGS},
                                       flagged=("auto_suspect", "sum"),
                                       suspects=("suspect_account", "sum")).reset_index()
    summary.insert(1, "bracket_name", summary["bracket"].map(MEDALS))
    tables = od.REPO_ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "account_screen_summary.csv", index=False)

    # --- report -----------------------------------------------------------------
    q = [0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
    dist = s[SIGNALS].quantile(q).T
    dist.columns = ["p1", "p5", "p50", "p95", "p99", "max"]
    dist["threshold"] = [f"> {HIGH[k]}" if k in HIGH else f"< {LOW[k]}" for k in SIGNALS]
    dist["flagged"] = [int(s["flag_" + k].sum()) for k in SIGNALS]
    print("signal distributions over", len(s), "players:")
    print(dist.round(3).to_string())
    print(f"(quiet_share missing for {s['quiet_share'].isna().sum()} players without a "
          f"{BLOCK_DAYS}-day block of {MIN_BLOCK}+ matches; party flags {int(s['flag_party'].sum())})")

    med = s.groupby("bracket")[SIGNALS].median()
    med.index = med.index.map(MEDALS)
    print("\nmedian per bracket:")
    print(med.round(3).to_string())

    print("\nplayers by number of signals:")
    print(s["n_signals"].value_counts().sort_index().to_string())
    strong = s[["flag_" + k for k in STRONG]].any(axis=1)
    print(f"flagged: {int(s['auto_suspect'].sum())} ({int(strong.sum())} by boost or throw days, "
          f"{int((s['n_signals'] >= MIN_SIGNALS).sum())} by {MIN_SIGNALS}+ signals); "
          f"suspects after hand check: {int(s['suspect_account'].sum())}")
    sizes = s["cv_group"].value_counts()
    print(f"\ncv groups: {len(sizes)} ({(sizes > 1).sum()} with 2+ players, "
          f"{sizes[sizes > 1].sum()} players in them, largest {sizes.max()})")
    print("\n" + summary.to_string(index=False))
    print(f"\nsaved {proc / 'account_screen.parquet'}\n      {proc / 'players.parquet'} (+ {', '.join(new_cols)})"
          f"\n      {interim / 'account_screen_check.csv'}\n      {tables / 'account_screen_summary.csv'}")
    return s


if __name__ == "__main__":
    run()

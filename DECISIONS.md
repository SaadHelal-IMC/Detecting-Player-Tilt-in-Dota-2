# Method decisions

Every method decision in this project and the reason behind it, grouped by
topic. This file is the basis for the Method chapter of the thesis.

## 1. Changes against the exposé

Two changes, pending confirmation by the supervisor:

1. **Global detectors, player-disjoint split.** The exposé said the detectors
   are trained per player and also evaluated on player-disjoint data, which
   cannot both hold. Mahalanobis stays per player (it is the baseline).
   Isolation Forest, one-class SVM and the LSTM autoencoder are trained once
   on the pooled baseline-deviation features of the training players and
   tested on held-out players. A per-player LSTM on a few hundred matches
   would not learn much.
2. **Outcome-free inputs plus a forward-looking check.** The proxy label uses
   loss streak, short requeue gap and performance drop, and the detectors see
   the gap and performance features, so agreement with the proxy is partly
   built in. Win, loss and streak features are therefore never detector
   inputs, and a second check uses information the detectors never see
   (section 8).

## 2. Data source (OpenDota API)

- Free tier: 60 calls per minute and about 3,000 per day.
- One call to `/players/{id}/matches` with `lobby_type=7`, `limit=500` and an
  explicit `project` field list returns a player's latest 500 ranked matches.
  All requested fields come back, including hero_damage, tower_damage and
  hero_healing.
- Only about 15% of matches are parsed (`version` set). The core study
  therefore uses summary-tier fields only; parsed features stay an extension.
- `party_size` is missing on about 30% of matches in the analysis window
  (mostly 2024 and 2025), so it is not used as a feature.
- Abandons (`leaver_status > 0`) are rare (2 of 500 matches on a test
  account), so they carry little weight in the forward check.
- Every raw response is cached locally, so the dataset can be rebuilt without
  new calls.

## 3. Player sampling

- `/publicMatches` with `min_rank` / `max_rank` returns the latest 100 public
  matches of a rank band, and most of them are not ranked (in a probe of the
  Legend band, 86 of 100 were normal lobbies, mostly Turbo, and 10 ranked).
  **Only ranked matches (lobby_type 7) are used as seeds**, so sampled
  players actually play ranked.
- `/matches/{id}` lists the 10 players; about 6 of 10 are public and show
  their rank_tier. Ranks inside one match can differ a lot (33 to 74 in a
  probed Turbo game), so **each candidate is assigned to the medal of their own
  rank_tier**, not to the band of the seed match.
- A candidate is **usable with at least 100 ranked matches inside the
  analysis window** (section 4).
- The medal with the fewest usable players is always served first, so thin
  medals get the effort. Target: 150 usable players per medal.
- Cost: about 3.3 API calls per usable player, most of it spent on private
  profiles, whose history call returns nothing. Nothing in the `/matches`
  player fields (personaname, computed_mmr, last_login, subscriber flags)
  predicts a private profile, so they cannot be skipped in advance.
- **Private profiles fall sharply with skill.** In a pilot of 20 usable
  players per medal, the candidates checked were (private vs. with data):
  Herald 72 vs. 25, Guardian 66 vs. 21, Crusader 43 vs. 21, Archon 31 vs. 20,
  Legend 42 vs. 20, Ancient 16 vs. 20, Divine 15 vs. 20, Immortal 2 vs. 20.
- **Final collection (2 to 7 Nov 2026):** 3,993 candidates checked for 1,200
  usable players. "Empty" means the history call returned no ranked match,
  which almost always means hidden match data; "too few" means fewer than 100
  ranked matches in the window.

  | Medal | Checked | Usable | Too few | Empty |
  | --- | --- | --- | --- | --- |
  | Herald | 997 | 15.0% | 10.1% | 74.8% |
  | Guardian | 751 | 20.0% | 6.1% | 73.9% |
  | Crusader | 552 | 27.2% | 3.8% | 69.0% |
  | Archon | 438 | 34.2% | 1.6% | 64.2% |
  | Legend | 368 | 40.8% | 0.3% | 59.0% |
  | Ancient | 344 | 43.6% | 0.9% | 55.5% |
  | Divine | 281 | 53.4% | 1.1% | 45.6% |
  | Immortal | 262 | 57.3% | 1.5% | 41.2% |

- **This is a stratified sample, not a simple random one.** Which players
  enter a medal depends only on which public ranked matches were recent at
  collection time and on the fixed rules above; no player was picked or
  removed by hand. Equal numbers per medal are on purpose, because SQ3
  compares medals and a simple random draw would be dominated by the middle
  medals. Results are reported per medal and not weighted to the player
  population.
- Limitations: lower-bracket samples are a more self-selected group (only
  about a quarter of those players expose their data), and sampling from
  recent public matches favours players who are active right now.

## 4. Analysis window

**Only ranked matches from 2024-11-01 on are used** (the two years before data
collection in November 2026).

- Players who rarely play ranked need years for 500 matches; some histories
  reach back to 2015. In the pilot, 2015 to 2020 had about 250 to 550 matches
  per year, 2021 to 2023 about 2,000 to 2,700, 2024 3,771, 2025 10,189 and
  2026 49,756.
- `average_rank` is missing on every match before 2022 (64% in 2022), and
  old matches span many patches.
- Effect on the pilot: 147 of 159 players keep at least 100 matches and
  60,156 of 73,714 rows remain. Losses are mostly in Herald and Guardian.

## 5. Bracket definition

**Bracket = the player's profile medal** (rank_tier shown in `/matches` at
sampling time), 8 medals from Herald to Immortal.

- Considered and rejected: the median lobby `average_rank` of the player's
  matches. It tops out at about 75 (Divine 5): for Immortal players the
  median is 75 and the 99th percentile 81, so in the pilot all 20 Immortal
  players landed in Divine. The mean rank_tier of a lobby also tends to read
  one medal low.
- Profile medal and the median lobby medal of the last 100 matches agree
  within one medal for 94% of players in the pilot and for 96% to 99% per
  bracket in the interim sample (section 9), so the profile is a fair
  description of where a player plays now.
- The lobby medal is kept in `players.parquet` as `lobby_medal` for
  robustness checks in SQ3.

## 6. Cleaning rules

Applied in this order by `src/clean/clean.py`:

1. duplicates of (player, match) dropped
2. non-ranked matches dropped
3. matches before the analysis window dropped
4. each history split into runs at breaks longer than 30 days; runs shorter
   than 20 matches dropped (a few isolated games say nothing about the
   current baseline)
5. players with fewer than 100 matches left dropped
6. games under 10 minutes flagged as `short_game` and kept

## 7. Features and baseline

| Decision | Choice |
| --- | --- |
| Baseline | causal robust z = (x − median) / (1.4826 × MAD) over the previous 50 to 100 matches; the first 50 matches are burn-in and never scored |
| Detector inputs | deviations of K, D, A, GPM, XPM, LH, hero damage, tower damage, duration; gap before the match (log); hero novelty |
| Not inputs | win, loss, streak counters (used only by the proxy label and the forward check) |
| Scoring unit | one score per match t; the LSTM-AE scores the 10-match window ending at t |
| Session | consecutive matches with a gap under 60 min; 30 and 120 min as sensitivity |

**Session cutoff from the gap distribution** (gap = start of a match minus
end of the previous one, interim sample of 771 players, 318,423 gaps):
gaps pile up at 4 to 12 minutes (queue and post-game; 25th percentile 8.3
min, median 25.8 min) and fall off steeply until about 30 to 40 minutes. On a
log scale the density is then flat from about 40 minutes to 8 hours, before
the next-day bump at 12 to 24 hours. The in-session pattern therefore ends
around 40 to 60 minutes, and 57% of all gaps are under 60 minutes. The few
negative gaps (0.07%, overlapping records) are set to 0.

## 8. Detectors and evaluation

| Decision | Choice |
| --- | --- |
| Model scope | Mahalanobis per player (Ledoit-Wolf shrunk covariance of the baseline window); Isolation Forest, OCSVM, LSTM-AE global |
| Split | 5-fold cross-validation grouped by `cv_group` (one player, or players who share at least 10 matches, section 10) and stratified by bracket; every player is in a test fold exactly once |
| Tuning | on a validation slice of players inside each training fold, never on the test fold |
| Threshold | top 5% of scores, set on the validation slice; AUPRC and P@k are threshold-free |

**Proxy label.** Match t is a weak tilt positive when all three hold:

1. matches t−1 and t−2 were both lost
2. the gap before t is below the player's own 25th percentile of gaps in the
   baseline window
3. the composite performance z of t (mean of GPM, XPM, LH, KDA, deaths
   sign-flipped) is below −1

Two variants run as sensitivity checks:

- **loose:** performance z below −0.5 instead of −1
- **same session:** as main, and t−2, t−1 and t all lie in one session

A strict variant (three losses, performance z < −1.5) was dropped: it fired
on only 98 of 242,094 scored matches, too few to evaluate.

Prevalence in the interim sample (771 players, 242,094 scored matches after
the 50-match burn-in):

| Label | Positives | Rate | Range across brackets |
| --- | --- | --- | --- |
| main | 1,514 | 0.63% | 0.51% to 0.82% |
| loose | 4,826 | 1.99% | 1.75% to 2.33% |
| same session | 881 | 0.36% | 0.21% to 0.54% |

The parts fire 26.1% (two-loss streak), 25.8% (gap under own 25th
percentile) and 7.5% (performance z < −1) of the time. Their product is
0.51%, a bit below the observed 0.63%, so the parts co-occur about 1.25
times more often than chance. A quarter of players never trigger the main
label; the median player has 2 positives.

Why 5-fold cross-validation instead of one 60/20/20 split: a single test set
would hold only about 300 main-label positives, roughly 40 per bracket, too
few for stable per-bracket results in SQ3. With every player tested once,
all 1,514 positives (about 190 per bracket) are used.

**Forward-looking check.** Per detector, flagged matches are compared with the
same player's unflagged matches on the win rate over the next 3 matches, the
abandon rate and whether the session ends. Paired Wilcoxon test per bracket.

## 9. Sample observations

From dataset v1 (1,180 players, 487,683 matches; collected 2 to 7 Nov 2026).
**v1 is frozen** in `frozen/v1/` in the data folder with a manifest of row
counts and SHA-256 checksums of every file and of the scripts that built
them (`python -m src.clean.freeze v1`). Features, labels and models read only
from v1; a changed dataset gets a new version, never an overwrite.

- 1,200 players passed the collector's check (at least 100 ranked matches in the
  window); 20 drop out in cleaning, because clean also removes runs under 20
  matches between 30-day breaks. Herald loses most (150 to 140), since
  low-bracket players take more breaks. Not topped up; CV folds are stratified
  by bracket.
- Low brackets play less: the median number of matches per player in the
  window is 279 (Herald) and 305 (Guardian), against 500 (the cap) from Archon
  upwards. The median span is 473 days in Herald against 256 in Immortal.
- The solo share, over matches with known party_size, is high in every bracket
  (0.65 to 0.81): highest in Herald (0.81) and Immortal (0.80), lowest in Legend
  and Ancient (0.65, 0.66). An earlier version counted missing party_size as
  not solo. Low brackets reach further back into 2024 and 2025, where
  party_size is mostly missing, so that produced a false rising trend (0.36 to
  0.69).
- The mean win rate rises with the bracket, from 0.45 in Herald to 0.52 in
  Immortal, probably because players sampled at a rank today are more often
  ones who climbed into it.
- Profile medal and recent lobby medal agree within one medal for 95% to 99%
  of players per bracket.
- 507 games under 10 minutes (flagged, not dropped).

## 10. Account screen

Bot farms reached ranked Dota in 2026, and a sold or shared account mixes two
people's play in one history. `src/clean/screen_accounts.py` computes eight
account-level signals per player from the cleaned matches (no API calls) and
marks a player as a suspect when at least two fire, or when one of the two
direct signals (boost, throw days) fires. **Suspects are kept**; the
sensitivity step reruns SQ2 without them.

| Signal | Flagged when | Players |
| --- | --- | --- |
| Quiet hours: share of matches in the quietest 6 hours of the day (UTC), median over 30-day blocks with 30+ matches | > 0.15 | 12 |
| Most matches started within 24 hours | > 27 | 11 |
| Spread: geometric mean of the robust CVs of GPM, duration and KDA | < 0.12 | 0 |
| Share of matches on the 3 most played heroes | > 0.95 | 12 |
| Party: 5-stack share (known party_size) or share of matches with another sampled player (seed matches left out) | either > 0.5 | 44 |
| Shift: change of heroes and play hours at one split point (min of the two TVDs, max over splits) | > 0.6 | 9 |
| Boost: best win rate over 100 matches in a row | > 0.75 | 15 |
| Throw days: UTC days with 8+ games, at most 20% wins and a median KDA under 0.5 | 1 or more | 7 |

- Thresholds sit at about the 99th percentile of the 1,180 players, so each
  signal alone flags about 1% (party about 4%, as two shares feed it). Spread
  is the exception: nobody comes close to scripted uniformity (lowest value
  0.29, a bot repeating identical games gives 0), so it keeps a fixed level and
  flags no one.
- **The thresholds were set after looking at this same sample, not fixed in
  advance.** A first run with provisional values (quiet hours over the whole
  window > 0.10, more than 24 matches in 24 hours, top-3 heroes > 0.90,
  5-stack share > 0.5 and shared matches > 0.2 as two separate signals, shift
  > 0.6, spread < 0.12) flagged 36 players, mostly ordinary ones; the changes
  are described in the next points. This is a data-driven choice. It is
  acceptable here because the screen only describes account quality, no
  account is removed from the main analysis, and everything was fixed and
  frozen (v1) before any tilt label or detector score existed. No decision
  about the data has used labels or detector results.
- Quiet hours are taken per 30-day block. Over the whole window, a player whose
  schedule moved (new job, semester) looks like someone playing around the
  clock: 168 players passed 0.10 that way, and their median per block is about
  0.10. Per block the highest value is 0.185, against 0.25 for play spread
  evenly over the day. 15 players have no block with 30+ matches and get no
  quiet-hours value.
- Boost and throw days each mark a suspect on their own, because they are
  direct evidence rather than unusual habits. Matchmaking keeps honest players
  near 50%: in 20,000 simulated histories of 500 games with a true win chance
  of 55%, a 100-game stretch above 75% came up in 0.035%, so fewer than one
  false flag is expected among 1,180 players. Both were added after the hand
  check below showed a deranking day that the other signals only caught by
  chance.
- 5-stack share and shared matches count as one signal. Both measure playing
  in a fixed group, and in a first run with separate counts they fired together
  for 10 players.
- Result: 1,083 players with no signal, 87 with one, 7 with two and 3 with
  three. **20 players (1.7%) are flagged**, all of them through boost or throw
  days: 15 with a boost stretch (all Ancient to Immortal, best stretches 0.76
  to 0.92) and 7 with throw days (12 days in all, Herald to Divine, mostly in
  5-stacks), 2 with both. **No player shows the bot pattern** (round-the-clock play together
  with very long days and uniform games). This fits the documented 2026 bot
  accounts having hidden match data, while only public profiles enter the
  sample (section 3).
- **Hand check (7 Nov 2026): both suspects confirmed, as smurf or boosted
  accounts rather than bots.** The Legend account wins 83% of its other games;
  on 9 Oct 2026 it lost 24 of 28 games in a 5-stack (median KDA 0.18, 183 GPM,
  18 minutes). The replay of one of them shows the whole team
  losing on purpose from minute 0: all five heroes walking into mid and
  feeding, wards placed all over the map and other plainly senseless play
  (deranking or a fixed match). The Ancient account won
  89% of its first 140 games and from 21 Feb 2026 plays other heroes at other
  hours with 45% wins, consistent with a boosted and then sold account
  (pattern only, no replay checked). Verdicts are in
  `account_screen_review.csv` in the data folder and override the automatic
  flag; flagged accounts without a verdict stay suspects.
- Throwing on purpose looks exactly like the tilt proxy label (losses in a
  row, short gaps, very poor performance), so deliberate throwing and extreme
  tilt are hard to tell apart from match statistics alone.
- Replays: a 29-day-old replay was still on Valve's server (HTTP 200 on the
  `replay_url` from OpenDota, 7 Nov 2026), so guides that give 7 to 14 days
  are wrong or outdated (no official retention period from Valve was found). The file is
  zstd-compressed despite its `.dem.bz2` name. The in-game download hung, so
  it was downloaded directly and decompressed into Dota's replays folder.
- Account ids with a suspect label stay in the data folder. The repo only holds
  counts per bracket (`results/tables/account_screen_summary.csv`).

**CV groups.** 3.8% of match rows (18,579) are in the histories of two or more
sampled players, so those players share outcomes and timing in them. Players
who share at least 10 matches are joined into one `cv_group` (126 players in 52
groups of 2 to 5; everyone else is a group of one), and the CV folds keep each
group together. Linking at a single shared match would chain 972 players into
one group, mostly through one-off games in the same rank pool.

## 11. Out of scope

- Parsed-replay features (ward placements, item timings, per-minute curves).

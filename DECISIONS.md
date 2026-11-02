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
| Split | 5-fold cross-validation grouped by player and stratified by bracket; every player is in a test fold exactly once |
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

From the cleaned interim sample (771 players, 319,194 matches; final numbers
follow with the frozen dataset):

- Low brackets play less: the median number of matches per player in the
  window is 280 (Herald) and 329 (Guardian), against 500 (the cap) from
  Archon upwards.
- The solo share rises with skill, from 0.36 in Herald to 0.69 in Immortal
  (with the party_size caveat from section 2).
- The mean win rate rises with the bracket, from 0.45 in Herald to 0.52 in
  Immortal, probably because players sampled at a rank today are more often
  ones who climbed into it.
- Profile medal and recent lobby medal agree within one medal for 96% to 99%
  of players per bracket.
- 416 games under 10 minutes.

## 10. Out of scope

- Parsed-replay features (ward placements, item timings, per-minute curves).

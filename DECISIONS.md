# Decision log

Running log of every method decision, newest at the bottom. This file feeds
the Method chapter, so each entry says what was decided and why.

## 2026-11-02 · Design locked before coding

Two changes against the exposé (to be confirmed with Prof. Ruiz Torrubiano):

1. **Global detectors, player-disjoint split.** The exposé said the detectors
   are trained per player and also evaluated on player-disjoint data, which
   cannot both hold. Mahalanobis stays per player (it is the baseline).
   Isolation Forest, one-class SVM and the LSTM autoencoder are trained once
   on the pooled baseline-deviation features of the training players and
   tested on held-out players. A per-player LSTM on ~200 matches would not
   learn much.
2. **Outcome-free inputs plus a forward-looking check.** The proxy label uses
   loss streak, short requeue gap and performance drop, and the detectors see
   the gap and performance features, so agreement with the proxy is partly
   built in. Win, loss and streak features are therefore never detector
   inputs, and a second check uses information the detectors never see.

| Decision | Choice |
| --- | --- |
| Model scope | Mahalanobis per player (Ledoit-Wolf shrunk covariance of the baseline window); IF, OCSVM, LSTM-AE global |
| Split | 60 / 20 / 20 train / validation / test by player, stratified by bracket |
| Baseline | causal robust z = (x − median) / (1.4826 × MAD) over the previous 50 to 100 matches; first 50 matches are burn-in, never scored |
| Detector inputs | deviations of K, D, A, GPM, XPM, LH, hero damage, tower damage, duration; gap before the match (log); hero novelty |
| Scoring unit | one score per match t; LSTM-AE scores the 10-match window ending at t |
| Session | consecutive matches with gap under 2 h (check in EDA) |
| Bracket | current profile rank_tier grouped into 8 medals; unranked dropped |
| History | latest 500 ranked matches (lobby_type 7) per player, at least 100 required |
| Threshold | top 5% of scores, set on validation players; AUPRC and P@k are threshold-free |

**Proxy label.** Match t is a weak tilt positive when all three hold:

1. matches t−1 and t−2 were both lost
2. the gap before t is below the player's own 25th percentile of gaps in the baseline window
3. the composite performance z of t (mean of GPM, XPM, LH, KDA, deaths sign-flipped) is below −1

Strict and loose variants are run as a sensitivity check.

**Forward-looking check.** Per detector, flagged matches are compared with the
same player's unflagged matches on the win rate over the next 3 matches, the
abandon rate (leaver_status > 0) and whether the session ends. Paired Wilcoxon
test per bracket.

**Out of scope for November:** parsed-replay features.

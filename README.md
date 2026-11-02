# Detecting Player Tilt in Dota 2

Bachelor thesis, Informatics, IMC Krems (supervisor: Prof. Rubén Ruiz Torrubiano).

Tilt is treated as a sustained shift away from a player's own behavioural
baseline across consecutive ranked matches. Match histories come from the
public OpenDota API, features are expressed as deviations from a causal
robust baseline, and four unsupervised detectors are compared (Mahalanobis,
Isolation Forest, one-class SVM, LSTM autoencoder). See `DECISIONS.md` for
every method decision.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q             # offline tests, no API calls
```

The repo sits in OneDrive, so keep the data folder outside it (the cache
grows to thousands of small files). Run this once, then open a new terminal:

```powershell
setx TILT_DATA_DIR C:\Users\User\tilt-data
```

Without it, data goes to `data/` inside the repo (ignored by git).

Optional: set `OPENDOTA_API_KEY` if you have a paid key. Without it the free
tier applies (about 60 calls per minute, 2,000 per day).

## Steps

| Step | Command | Output |
| --- | --- | --- |
| Check fields on one account | `python -m src.collect.check_fields <steam32_id>` | `data/interim/own_matches_check.csv` |

More steps are added as they are built (collect, clean, features, labels,
models, eval, analysis).

## Layout

```
data/            raw API cache and derived tables (not in git, rebuilt by scripts)
src/collect/     OpenDota client and collection scripts
tests/           offline tests
notebooks/       EDA only, nothing the results depend on
results/         figures and tables for the thesis
DECISIONS.md     method decision log
```

Every raw API response is cached under `data/raw/`, so any collection run can
be stopped and restarted without losing calls.

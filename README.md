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

By default all data goes to `data/` inside the repo (ignored by git). The
API cache grows to thousands of small files, so if the repo lives in a synced
folder (OneDrive, Dropbox), point `TILT_DATA_DIR` to a folder outside it.
On Windows, run this once and then open a new terminal (restart VS Code if
you use its terminal):

```powershell
setx TILT_DATA_DIR C:\path\to\tilt-data
```

Optional: set `OPENDOTA_API_KEY` if you have a paid key. Without it the free
tier applies (60 calls per minute, about 3,000 per day).

## Steps

Output paths are relative to the data folder, except `results/`, which is in
the repo.

| Step | Command | Output |
| --- | --- | --- |
| Check fields on one account | `python -m src.collect.check_fields <steam32_id>` | `interim/own_matches_check.csv` |
| Probe the sampling endpoints | `python -m src.collect.probe_sampling [min_rank max_rank]` | printed only |
| Sample players and download histories (resumable) | `python -m src.collect.collect [--target 150] [--max-calls N]` | `interim/candidates.csv`, `raw/` |
| Clean into one tidy table (cache only, no API) | `python -m src.clean.clean` | `processed/matches.parquet`, `processed/players.parquet`, `results/tables/dataset_summary.csv` |
| Screen accounts for bots and shared accounts (after clean) | `python -m src.clean.screen_accounts` | `processed/account_screen.parquet`, `players.parquet` (+ `suspect_account`, `cv_group`), `interim/account_screen_check.csv`, `results/tables/account_screen_summary.csv` |
| Freeze the dataset as a fixed version (later steps read only this) | `python -m src.clean.freeze v1` (`--check` verifies it) | `frozen/v1/` with `manifest.json` (rows, SHA-256 of files and scripts, git commit) |

Still to come: features, labels, detectors, evaluation and analysis.

## Layout

```
data/            raw API cache and derived tables (not in git, rebuilt by scripts)
data/frozen/     fixed dataset versions with checksum manifests (not in git)
src/collect/     OpenDota client and collection scripts
src/clean/       raw cache -> tidy match and player tables
tests/           offline tests
notebooks/       EDA only, nothing the results depend on
results/         figures and tables for the thesis
DECISIONS.md     method decisions and their reasons
```

Every raw API response is cached under `raw/` in the data folder, so any
collection run can be stopped and restarted without losing calls.

## Note on tools

Parts of the code in this repository were written with the help of an AI
coding assistant (Claude). All method decisions and results were reviewed by
the author.

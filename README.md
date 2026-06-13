# FIFA 2026 World Cup Predictor

ML-powered predictions for the 2026 FIFA World Cup — match outcomes, expected
goals/scorelines, player impact, and full Monte Carlo tournament simulations,
served through a FastAPI backend and a React frontend.

**Live demo:** https://fifa-2026-predictor-alpha.vercel.app/

---

## What it does

- **Match predictor** — pick any two of the 48 qualified teams and get
  win/draw/loss probabilities (XGBoost classifier), expected goals, and the
  most likely scoreline (Bivariate-Poisson model).
- **Team pages** — Elo trajectory, recent form, squad list, and tournament
  path probabilities for each team.
- **Player pages** — searchable player database with position-aware
  composite quality scores (xG, club form, market value, experience).
- **Bracket simulator** — runs a full live simulation of the official WC 2026
  bracket (group stage → R32 → R16 → QF → SF → Final) and animates the result
  round by round.
- **Tournament odds** — aggregate progression probabilities for all 48 teams
  from a 100,000-run Monte Carlo simulation.

## Tech stack

| Layer        | Tech |
|--------------|------|
| Data pipeline | Python, pandas, requests/BeautifulSoup (football-data.org, API-Football, FBref, Understat, StatsBomb, Transfermarkt) |
| Modeling      | XGBoost (W/D/L classifier), Bivariate-Poisson scoreline model, Elo ratings |
| Simulation    | Custom Monte Carlo engine implementing the official FIFA WC 2026 bracket (Matches 73-104, incl. Annex C third-place rules) |
| Backend       | FastAPI, served with Uvicorn |
| Frontend      | React + Vite, React Router, Recharts |

---

## Quick start (run it locally)

The repo ships with the processed datasets and a trained model already
included (`data/processed/`, `models/`), so you can run the full app
**without** an API key or re-running the data pipeline.

### 1. Backend (FastAPI)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-api.txt   # lightweight, runtime-only deps
uvicorn src.api.main:app --reload
```

The API is now running at `http://localhost:8000` (interactive docs at
`http://localhost:8000/docs`).

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend talks to the API at
`http://localhost:8000` by default (configurable via `VITE_API_URL`, see
[Configuration](#configuration)).

---

## Rebuilding the dataset / running your own simulation

If you want to refresh the data, retrain the model, or run your own Monte
Carlo simulation from scratch:

```bash
# 1. Install full dependencies (pipeline + training + notebooks)
pip install -r requirements.txt
cp .env.example .env   # add API keys for football-data.org / API-Football (optional sources)

# 2. Run the data pipeline — fetches & processes results, Elo, squads,
#    player stats (FBref, Understat, StatsBomb, Transfermarkt), WC2026
#    groups/fixtures into data/processed/
python -m src.data_pipeline.run_pipeline

# 3. Build/refresh feature tables and retrain the model
python -m src.models.train

# 4. Run the 100k-run Monte Carlo tournament simulation
#    -> writes data/processed/monte_carlo_results.csv
python -m src.simulation.monte_carlo
```

Re-running step 4 also regenerates `data/processed/prediction_cache.pkl`,
which the `/api/simulation/run` endpoint uses so live "what-if" bracket runs
are instant instead of taking ~40s on first request.

> Note: some pipeline steps fetch live data and may need API keys (see
> `.env.example`) and can take a while due to scraping rate limits.

---

## Configuration

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `VITE_API_URL` | frontend (Vite env) | `http://localhost:8000` | Base URL the frontend uses to call the API |
| `ALLOWED_ORIGINS` | backend (process env) | `http://localhost:3000` | Comma-separated list of origins allowed by CORS |
| `FOOTBALL_DATA_API_KEY`, `API_FOOTBALL_KEY` | backend pipeline | — | Optional, only needed for `src.data_pipeline.run_pipeline` |

---

## Deploying your own instance

This repo is set up to deploy as two services:

- **Backend** → [Render](https://render.com) — `render.yaml` is included as a
  Blueprint. Connect the repo, deploy, then set `ALLOWED_ORIGINS` to your
  frontend's URL.
- **Frontend** → [Vercel](https://vercel.com) — set the project root to
  `frontend/`, and add an env var `VITE_API_URL` pointing at your Render
  backend URL (e.g. `https://fifa-2026-predictor-api.onrender.com`).

Once both are live, update the **Live demo** link at the top of this README (already set above for the deployed instance).

---

## Project layout

```
src/
  data_pipeline/   # fetch & process raw data into data/processed/
  features/        # feature engineering (team, player, match-level)
  models/          # XGBoost outcome model + scoreline model + training
  simulation/       # bracket logic + Monte Carlo tournament simulation
  api/             # FastAPI app and routers
data/processed/    # cleaned CSVs + cached predictions consumed by the API
models/            # trained model artifacts (.pkl)
frontend/          # React + Vite UI
notebooks/         # exploratory analysis
```

## Notes

- The knockout bracket in `src/simulation/bracket.py` implements the
  **official FIFA WC 2026 schedule** (Matches 73-104), including the
  Annex C third-place slot allocation, so simulated paths match the real
  tournament routing.
- Player composite scores are **position-aware** (see
  `player_features.POSITION_WEIGHTS`): attackers are scored mainly on xG
  signals, defenders/keepers on market value + experience.
- FIFA ranking points (`src/features/team_features.py`) are a hardcoded
  snapshot — see `FIFA_RANKING_LAST_UPDATED` for the date, and refresh
  periodically from the official FIFA rankings.
- Team naming is canonicalized to match `data/processed/wc2026_groups.csv` /
  `wc2026_fixtures.csv` (e.g. "Côte d'Ivoire", "Curaçao"); other data sources
  are aliased to this spelling at lookup time.

## License

See [LICENSE](LICENSE).

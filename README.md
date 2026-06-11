# FIFA 2026 World Cup Predictor

ML-powered predictions for the 2026 FIFA World Cup: match outcomes, expected
goals/scorelines, player impact, and full Monte Carlo tournament simulations.

## How it works

1. **Data pipeline** (`src/data_pipeline/`) — fetches and processes historical
   international results, Elo ratings, squads, player stats (FBref,
   Understat, StatsBomb, Transfermarkt), and the WC 2026 groups/fixtures.
   Run end-to-end with:

   ```bash
   python -m src.data_pipeline.run_pipeline
   ```

2. **Feature engineering** (`src/features/`) — builds per-match features
   (Elo differential, recent form, head-to-head record, tournament
   experience, FIFA ranking points, squad/player quality) used by the models.

3. **Models** (`src/models/`)
   - `match_outcome.py` — XGBoost multiclass classifier predicting
     win/draw/loss probabilities.
   - `match_outcome.predict_score()` — lightweight Bivariate-Poisson-style
     model for expected goals and the most likely scoreline.

   Train (or retrain after data/feature changes) with:

   ```bash
   python -m src.models.train
   ```

4. **Tournament simulation** (`src/simulation/`) — `bracket.py` encodes the
   WC 2026 group draw and knockout bracket; `monte_carlo.py` runs the full
   tournament thousands of times to produce per-team progression
   probabilities, written to `data/processed/monte_carlo_results.csv`.

   ```bash
   python -m src.simulation.monte_carlo
   ```

5. **API** (`src/api/`) — FastAPI backend serving predictions and simulation
   results to the frontend.

   ```bash
   uvicorn src.api.main:app --reload
   ```

6. **Frontend** (`frontend/`) — React + Vite app that consumes the API.

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys for optional data sources
```

## Project layout

```
src/
  data_pipeline/   # fetch & process raw data into data/processed/
  features/        # feature engineering (team, player, match-level)
  models/          # XGBoost outcome model + scoreline model + training
  simulation/      # bracket logic + Monte Carlo tournament simulation
  api/             # FastAPI app and routers
data/processed/    # cleaned CSVs consumed by features/models/api
models/            # trained model artifacts (.pkl)
frontend/          # React + Vite UI
```

## Notes

- FIFA ranking points (`src/features/team_features.py`) are a hardcoded
  snapshot — see `FIFA_RANKING_LAST_UPDATED` for the date, and refresh
  periodically from the official FIFA rankings.
- Team naming is canonicalized to match `data/processed/wc2026_groups.csv` /
  `wc2026_fixtures.csv` (e.g. "Côte d'Ivoire", "Curaçao"); other data sources
  are aliased to this spelling at lookup time.

# FIFA 2026 Predictor — Data & Logic Review

Reviewed: 2026-06-10 (one day before WC2026 kicks off). Scope: `data/`, `src/features`, `src/models`, `src/simulation`, `src/api`, plus the ad-hoc `fix_*.py` patch scripts.

## 1. Data model overview

The pipeline is: `fetch_*` scripts → `data/processed/*.csv` → `src/features` (team + player feature builders) → `src/models/match_outcome.py` (XGBoost W/D/L classifier + Bivariate-Poisson scoreline) → `src/simulation` (group stage + bracket Monte Carlo, 100k runs) → `src/api` (FastAPI, read-only over the processed CSVs) → React frontend.

Core entities:
- **Teams** (48, groups A-L) — `wc2026_groups.csv`, `wc2026_fixtures.csv` (104 matches), `elo_ratings.csv` (Elo + momentum + confederation), hardcoded `FIFA_RANKING_POINTS` dict.
- **Players** (1,245 in `wc2026_squads.csv`) — joined against StatsBomb (intl xG), Understat + FBref (club form), Transfermarkt (market value) into `player_scores.csv` (composite score = 0.45·xG + 0.35·club form + 0.20·experience).
- **Historical matches** (21,458 since 2004) — `international_results.csv`, used for Elo momentum, recent form, H2H, and model training.
- **Outputs** — `monte_carlo_results.csv` (per-team advancement probabilities), served by the API.

The group draw in `wc2026_groups.csv` / `bracket.py` matches the **actual official FIFA draw from Dec 5, 2025** ([source](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_draw)) — that part is correct and current.

## 2. Critical logic bugs

### 2.1 Train/serve skew on `fifa_points_norm_diff_sq` (high impact)
`TRAINING_FEATURES` in `src/models/match_outcome.py` includes `fifa_points_norm_diff_sq`, and it's computed correctly during training (`_build_training_row`). But `build_match_features()` in `src/features/match_features.py` never computes this field. At inference time `predict()` does `feats.get("fifa_points_norm_diff_sq", 0.0)` → **always 0.0**, while the model was trained on real (non-zero, often large for mismatched teams) values. This silently distorts every live prediction for any match with a meaningful FIFA-ranking gap — which is most knockout matches.

**Fix:** add the same `diff * abs(diff)` computation to `build_match_features()`.

### 2.2 `h2h_win_rate_b` never exists → `h2h_advantage` is a redundant constant offset
`get_h2h_features()` only ever returns `h2h_win_rate_a`. Both training (`match_outcome.py`) and inference (`match_features.py`) compute `h2h_advantage = h2h_win_rate_a - h2h.get("h2h_win_rate_b", 0.33)`, so `h2h_advantage` is just `h2h_win_rate_a − 0.33` everywhere — a wasted feature slot that's perfectly collinear with `h2h_win_rate_a`. (Consistent train/serve, so not a skew bug, just dead signal.)

**Fix:** either drop `h2h_advantage`, or make it meaningful — e.g. compute team B's H2H win rate directly (`1 − win_rate_a − draw_rate`, or symmetric lookup) so it captures something `h2h_win_rate_a` doesn't.

### 2.3 Dead/wrong-key features in `match_features.py`
`goals_scored_per_game_a/b` and `goals_conceded_per_game_a/b` read `form_a.get("goals_scored_per_game", 1.2)`, but `get_team_recent_form()` actually returns `goals_scored_pg` / `goals_conceded_pg`. These four fields are always the hardcoded defaults (1.2 / 1.2 / 1.2 / 1.0). They aren't in `TRAINING_FEATURES`/`INFERENCE_FEATURES` so the model is unaffected, but it's dead, misleading code that looks like a working feature.

**Fix:** either fix the key names and wire them into `INFERENCE_FEATURES`/retrain (form-derived goals are useful signal), or delete the dead block.

### 2.4 API ignores the actual Poisson-derived scoreline
`src/api/routes/match.py` does `score_result.get("score_a", round(xg_a))` / `score_a`/`score_b`, but `predict_score()` returns `likely_score_a`/`likely_score_b`. So the API **always** falls back to naive `round(xg)` for "most likely scoreline", while `score_prob` (the probability) is computed for the *actual* Poisson-argmax score that's never returned. The displayed score and its displayed probability can therefore refer to two different scorelines.

**Fix:** read `likely_score_a`/`likely_score_b`.

### 2.5 Team-name mismatches across data sources (Côte d'Ivoire / Ivory Coast, Curaçao)
Four different spellings are used across the codebase for the same two teams:

| Source | Côte d'Ivoire spelling | Curaçao spelling |
|---|---|---|
| `wc2026_groups.csv` / `wc2026_fixtures.csv` | `Côte d'Ivoire` | `Curaçao` |
| `wc2026_squads.csv` | `Ivory Coast` | `Curaçao` |
| `elo_ratings.csv` (processed) | `Côte d'Ivoire` | `Curacao` (no cedilla) |
| `bracket.py` `GROUPS` dict (hardcoded) | `Ivory Coast` | `Curaçao` |
| `FIFA_RANKING_POINTS` dict | `Ivory Coast` | `Curaçao` |
| `TEAM_NAME_ALIASES` | maps `Ivory Coast` → `Côte d'Ivoire`, `Curaçao` → `Curacao` (only applied in `team_features.py` Elo/form lookups) |

This works *by accident* for the Monte Carlo path (which uses the hardcoded `bracket.GROUPS`, spelled "Ivory Coast"/"Curaçao", matching squads + FIFA dict + alias-resolved Elo). But `build_features_from_fixtures()` (used for the actual fixture list) reads team names straight from `wc2026_fixtures.csv`, where Côte d'Ivoire is spelled with the accented form. For that code path:
- `get_fifa_ranking_feature("Côte d'Ivoire")` → key miss → silently returns the **default 1400.0** instead of the real ~1555-1676 value.
- `get_team_player_features("Côte d'Ivoire")` → no match in `wc2026_squads.csv` (which says "Ivory Coast") → falls back to generic average squad features.

Net effect: any prediction generated via the fixtures-based path under-rates Côte d'Ivoire specifically.

**Fix:** pick **one** canonical spelling (recommend the official FIFA spelling, "Côte d'Ivoire" / "Curaçao", since that's what the live fixtures/groups data uses), normalize every CSV and dict to it at load time via a single shared alias table, and delete the three duplicate copies of `TEAM_NAME_ALIASES`/`FIFA_RANKING_POINTS` (see 3.2).

### 2.6 Duplicate player rows from name-collision joins
`player_scores.csv` has 1,247 rows vs 1,245 in `wc2026_squads.csv` — two players (`Théo Bongonda`, `José Luis Rodríguez`) appear twice. `_norm_name()` reduces names to "first-word + second-word, lowercased" for joining StatsBomb/Understat/FBref/Transfermarkt onto the squad list. When the lookup table itself contains two distinct people who reduce to the same `name_key` (common names, or a player who appears under two transliterations), the `how="left"` merge becomes one-to-many and **duplicates the squad row**, double-weighting that player in `get_team_player_features()`'s top-5 attacker average.

**Fix:** dedupe each lookup table on `name_key` *before* merging (e.g., keep the row with the most matching club/competition data, or require an exact-name + team/nationality match), and assert `len(df) == len(squads)` after the joins in `build_player_quality_table()`.

## 3. Data freshness & external-source issues

### 3.1 Hardcoded FIFA ranking points are now stale — and the ranking order has flipped
`FIFA_RANKING_POINTS` in `team_features.py` is a static dict labeled "April 2026", with **France #1 (1877.32), Spain #2, Argentina #3**. As of the **June 6, 2026 live FIFA ranking** (next official update June 11 — the day the tournament starts), **Argentina has retaken #1**, ahead of Spain and France ([source](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/top-ranked-teams-france-spain-argentina-england-portugal)). Since `fifa_points_norm_*` and `fifa_points_norm_diff_sq` are explicitly described as some of the strongest features, every Argentina match (including their group-J opener) is being scored with an outdated relative-strength signal — right when it matters most.

**Fix:** don't hardcode a point-in-time snapshot. Either (a) fetch the live ranking via a small scraper/API call as part of `run_pipeline.py` and cache it with a date stamp, or at minimum (b) refresh the dict from the June 11 update before the tournament starts, and add a `last_updated` field that the pipeline warns about if it's >30 days old.

### 3.2 Three different "team strength" signals disagree, and the codebase doesn't reconcile them
- `elo_ratings.csv`: Spain #1 (2165), Argentina #2 (2113).
- `FIFA_RANKING_POINTS` (stale, April): France #1, Spain #2, Argentina #3.
- Live FIFA ranking (June): Argentina #1, Spain #2, England #4, Brazil #6.

The model uses both Elo and FIFA points as separate features, which is reasonable in principle (they capture different things), but nothing validates that they're *currently consistent* — and right now the FIFA feature is simply outdated. Worth a periodic cross-check / sanity report comparing the two rankings and flagging large divergences.

### 3.3 `international_results.csv` starts in 2004 — fine, but confirm `tournament_type` taxonomy is current
Training restricts to `["world_cup", "continental_final", "nations_league"]`. This is a reasonable, well-justified choice (documented in `train.py`), but worth double-checking that recent 2025/2026 Nations League and continental-final fixtures (UEFA Nations League finals June 2025, etc.) were correctly classified — a misclassified recent match would silently drop out of training.

## 4. Code quality / maintainability

### 4.1 `team_features.py` has the same ~30-line block (TEAM_NAME_ALIASES, FIFA_RANKING_POINTS, get_fifa_ranking_feature) duplicated **three times** (lines 14-41, 44-71, 75-102)
This is a direct artifact of the `fix_*.py` regex-patch scripts (`add_fifa_ranking.py`, `fix_aliases_clean.py`, `fix_fifa_weight.py`, etc.) being applied without checking whether the target block already existed. Functionally harmless (last definition wins), but it's a landmine — a future edit to one copy won't apply to the others, and it roughly triples the maintenance surface for the alias table that's already a source of bugs (2.5).

**Fix:** collapse to a single definition; delete the dead copies.

### 4.2 The 13 `fix_*.py` / `add_fifa_ranking.py` scripts at the project root
These are one-off, already-applied string-replacement patches (several literally search-and-replace specific code blocks). They're no longer needed, clutter the root, and risk being re-run accidentally (most have `if old in f` guards, but a couple don't). Recommend deleting them now that their changes are baked into `src/`, and instead capturing the *lessons* (the bugs in §2) as proper code fixes + tests.

### 4.3 Stray artifacts
`Mbappe`, `hi.py`, `streak.md`, `downloaded_files/driver_fixing.lock` are all 0-byte or scratch files left in the repo root — safe to remove. `notebooks/*.ipynb` and `README.md` are all empty (0 bytes) — the project has no documentation of how to run the pipeline/API/frontend.

### 4.4 CORS config
`allow_origins=["http://localhost:3000", "*"]` combined with `allow_credentials=True` — browsers reject `*` with credentials, so this either does nothing useful or (depending on FastAPI/Starlette version) silently behaves like `*` without credentials. Pick one: either lock to specific origins (recommended for anything beyond local dev) or drop `allow_credentials`.

## 5. Recommended priority order

1. **Fix 2.1 (`fifa_points_norm_diff_sq` train/serve skew)** and **2.5 (name normalization)** — these directly corrupt live predictions, and 2.5 specifically hurts a team in a live group (Group E: Côte d'Ivoire/Curaçao) right before kickoff.
2. **Refresh FIFA ranking points (3.1)** — the June 11 official update lands the same day as the tournament opener; the model is currently using April data with the wrong #1 team.
3. **Fix 2.6 (duplicate player rows)** — affects squad attack scores for at least DR Congo and Panama.
4. **Fix 2.4 (API scoreline key mismatch)** — low effort, visible to end users.
5. Clean up 2.2/2.3 (dead features) — low risk, improves model quality on next retrain.
6. Housekeeping (4.1-4.4) — collapse duplicated blocks, delete `fix_*.py`/scratch files, write a real README, fix CORS.

After fixing 2.1, 2.5, 2.6 and refreshing FIFA points, **retrain the XGBoost model** (`python -m src.models.train`) and **re-run the Monte Carlo simulation** (`python -m src.simulation.monte_carlo`) so `monte_carlo_results.csv` (and everything the API/frontend serves) reflects the corrected pipeline.

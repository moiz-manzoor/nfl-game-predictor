# NFL Game Predictor

Predicting NFL game winners using pre-game team performance data, built on 10 seasons of historical data via `nflreadpy`.

---

## Overview

This project predicts NFL game winners using only information available before kickoff: rolling team performance stats (EPA, success rate, point differential, turnovers, and more), each computed strictly from prior games to avoid leakage. Two logistic regression models were trained on different time-based splits and evaluated against Vegas closing lines as a real-world benchmark. The pipeline also supports generating predictions for genuinely upcoming, unplayed weeks, not just backtesting historical ones.

**Key result:** Model A reaches 64.4% accuracy (0.618 log loss) on the 2025 season, compared to a Vegas benchmark of 66.2% accuracy (0.607 log loss) on the same games. The model tracks the market closely without using betting lines as an input.

---

## Table of Contents

- [Motivation](#motivation)
- [Data](#data)
- [Feature Engineering](#feature-engineering)
- [Methodology](#methodology)
- [Results](#results)
- [Vegas Benchmark Comparison](#vegas-benchmark-comparison)
- [Weekly Prediction Pipeline](#weekly-prediction-pipeline)
- [Production Model](#production-model)
- [Limitations](#limitations)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Future Work](#future-work)

---

## Motivation

Built to demonstrate an end-to-end sports analytics pipeline: pulling raw data, engineering leakage-safe features, training and tuning a model, and benchmarking it against the best available real-world predictor (the betting market), extending prior work in sports-focused modeling (MLB pitch prediction, NBA schedule analysis) into a new domain: full game outcome prediction with genuine forward-looking deployment, not just a one-time backtest.

---

## Data

- **Source:** [`nflreadpy`](https://github.com/nflverse/nflreadpy) (Python port of nflverse's `nflreadr`)
- **Scope:** 2016-2025 for team stats and play-by-play (10 completed seasons); schedules extended through 2026 to support predicting upcoming, unplayed weeks
- **Raw inputs:** Play-by-play data (used to compute EPA, success rate, and all Tier 2 stats), schedules (rest days, home/away, moneylines, divisional flags)
- **Data quality notes:** Team stats data lacks a unified offensive/defensive EPA split, so play-by-play is used directly to compute EPA and success rate per team-week using `posteam`/`defteam`. A franchise relocation naming mismatch (Raiders: OAK→LV, Chargers: SD→LAC) between play-by-play and schedule data was identified and corrected, since it silently dropped rows for both franchises' historical games before the fix.

---

## Feature Engineering

Every feature is a backward-looking, shifted rolling aggregate computed strictly from games prior to the one being predicted. This is the central design constraint of the project: a team's Week 6 features only ever reflect Weeks 1-5.

**Leakage prevention rule:** the modeling table only includes `_last3`, `_last5`, and `_s2d` (season-to-date) rolling differentials, computed via `shift(1)` before the rolling/expanding window. Raw, same-game differentials (e.g. `off_epa_diff`) reflect the outcome of the game being predicted and are deliberately excluded from the feature set, even though they're computed for reference.

**Tier 1 features:**
- Offensive EPA/play, Defensive EPA/play allowed
- Offensive/defensive success rate
- Rolling point differential
- Rest days, home field

**Tier 2 features:**
- Turnover margin
- Third-down conversion rate
- Explosive play rate (10+ yd runs, 15+ yd passes, PFF thresholds)
- Red zone touchdown rate
- Sack rate
- Pace (plays per game)
- Divisional game, short week, and bye-week flags

**Note:** due to a feature-selection bug found during v2 development (see Limitations), Model A and Model B were actually trained on 34 of these features, not all 39. The divisional game, short week, and bye-week flags above were computed and saved into `modeling_table.csv`, but a naming-convention bug excluded them from training. This is disclosed rather than corrected retroactively, to keep the reported 64.4%/64.7% numbers reproducible; the fix is applied going forward starting with the production model.

All rolling stats use last-3, last-5, and season-to-date windows, giving the model both a recency-weighted and a stability-weighted view of each team's form.

**Leakage testing:** `tests/test_features.py` independently recalculates specific EPA-based rolling stats (e.g. `off_epa_last5`, `def_epa_s2d`) directly from raw play-by-play for individual team-weeks and asserts they match the values produced by the pipeline, verifying the shift/rolling logic is correct for those features rather than assuming it. Coverage does not currently extend to the Tier 2 stats (turnover margin, third-down rate, sack rate, explosive rate, red zone rate).

---

## Methodology

- **Model:** Logistic regression (`scikit-learn`), chosen as an interpretable baseline that produces directly readable coefficients for feature importance analysis, before adding model complexity. A random forest comparison on the same features/target is planned (see Future Work) to test whether a more flexible model captures non-linear interactions logistic regression misses. Features are standardized via `StandardScaler` before training.
- **Training data:** 2,751 games total (2016-2025), after dropping ties
- **Features actually used:** Model A and Model B were trained on 34 features, not the 39 described above, due to a feature-selection bug found during v2 development (see Feature Engineering note and Limitations)
- **Regularization:** C=0.01, selected via a regularization strength sweep evaluated on each split's own test set. This means the sweep saw the same data used for final reporting, a stricter setup would tune C on a separate validation period (e.g. holding out 2023 from training) and report results on a completely untouched test set. Flagged here rather than corrected, see Limitations.
- **Two time-based splits, trained independently:**
  - **Model A:** trained on seasons ≤2024, tested on 2025
  - **Model B:** trained on seasons ≤2023, tested on 2024-2025
- **Target:** `home_team_win` (binary). Tied games (~0.4% of the data) are dropped since a binary target can't represent a tie.
- **Missing rolling data** (Week 1 of each season, or 2016's earliest weeks) is filled with 0, treated as "no signal yet" rather than a real team-quality assumption.
- **Multicollinearity:** the `_last3`, `_last5`, and `_s2d` versions of each stat are, by construction, correlated with each other. L2 regularization (built into logistic regression) helps limit any single version from dominating, but this wasn't independently tested via variance inflation factor or similar diagnostics.

---

## Results

| Model | Test set | Accuracy | Log loss |
|---|---|---|---|
| Always pick home team | 2025 | ~57-58%* | — |
| Model A | 2025 (true holdout) | 64.4% | 0.618 |
| Model B | 2024-2025 (true holdout) | 64.7% | 0.623 |
| Vegas (2025) | 2025 | 66.2% | 0.607 |

*Historical NFL home-win rate; not recomputed on this exact dataset, included as a reference point rather than an exact benchmark.

Model A is trained on seasons ≤2024, so 2025 is its only genuine out-of-sample test set. Model B is trained on seasons ≤2023, making its full 2024-2025 test range a true holdout as well. Running the `--backtest` flag with `--model a` will still process both 2024 and 2025 games for convenience, but note that 2024 is in-sample for Model A specifically, its headline result is the 2025-only number above.

Both models sit a few points behind the market, which is expected and reflects a realistic ceiling for a model built purely on team-level rolling stats without injury reports, QB-specific data, or line movement. The 2025 test set is 284 games; a McNemar's test on the paired Model A vs Vegas outcomes (statistic=0.29, p=0.59, see Limitations) confirms the ~2-point gap is not statistically significant at this sample size, so it should be read as directional rather than confirmed.

**Top features by coefficient magnitude (Model A):** rolling point differential (last5), season-to-date offensive success rate, and rolling offensive success rate (last5) were the strongest predictors, consistent with football intuition, recent scoring margin and offensive efficiency carry real signal. Note that rolling point differential is mechanically close to "which teams have recently been winning," so part of its predictive power likely reflects recent win/loss outcomes rather than purely situational skill. Given the multicollinearity between each stat's `_last3`/`_last5`/`_s2d` versions (see Methodology), individual coefficient magnitudes should be read as suggestive rather than a precise ranking, the correlated versions can trade off importance with each other in ways that don't reflect the underlying stat's true predictive value.

Feature importance charts, confusion matrices, and calibration plots are saved in `reports/`.

---

## Vegas Benchmark Comparison

Vegas closing moneylines are converted to de-vigged implied probabilities (removing the sportsbook's built-in margin) and compared against Model A's predicted probabilities on the same 2025 test set. This comparison is specific to Model A; Model B was not separately benchmarked against Vegas in `train_classifier.py`, though `combine_predictions.py` does include Vegas implied probabilities alongside both models' predictions for any given week. Vegas lines are used strictly as an evaluation benchmark, never as a training feature, since the point of this project is to see how far pre-game team statistics alone can get without using the market's own information.

**Important caveat:** this isn't a fully apples-to-apples comparison. Vegas closing lines incorporate information released right up to kickoff, final injury reports, weather, lineup news, and sharp money movement, while the model's features are frozen days earlier. The gap to Vegas partly reflects missing feature categories (QB continuity, injuries) and partly reflects a genuinely later information cutoff on the market's side.

The calibration overlay (`reports/calibration_comparison.png`) shows Model A's probability estimates tracking the market's closely across nearly the full probability range, with Vegas maintaining a consistent edge in both accuracy and calibration.

---

## Weekly Prediction Pipeline

Beyond backtesting, the pipeline supports generating real predictions for upcoming, unplayed weeks. Run `python src/data_pipeline.py` first to refresh the raw data, moneylines and schedules update as game time approaches, and a new week's games won't appear until they're pulled:

```bash
python src/predict_week.py --model a --season 2026 --week 1
```

This rebuilds all features using only data available as of the target week (correctly falling back to the end of the prior season for Week 1, when no in-season history exists yet), loads the trained model, and outputs win probabilities alongside Vegas implied probabilities for comparison.

`track_season.py` extends this into a running accuracy tracker: it checks which weeks of the current season have actually been played, re-predicts each of them, and maintains a season-to-date accuracy record for Model A against Vegas. `combine_predictions.py` merges Model A and Model B's predictions for the same week into one side-by-side comparison file.

---

## Production Model

Model A and Model B (above) are evaluation models. Their holdout seasons (2025, and 2024-2025 respectively) exist specifically to produce trustworthy, reported accuracy numbers, the 64.4%/64.7% figures in this README depend on those games never being seen during training.

Once a season is fully complete, though, holding it out forever stops making sense for a model meant to actually predict future games. `train_production_model.py` trains a separate model on every available season (2016-2025) with no holdout, specifically for live 2026 predictions. Same regularization (C=0.01) as Model A/B.

**Feature set difference from Model A/B:** the production model uses 39 features (`FEATURE_COLS` in `features.py`), an explicit list rather than the naming-convention selection (`endswith('_diff')`) that Model A/B use. That naming-convention selection was found to be a bug during v2 development, it silently excluded 5 documented features (`div_game`, `home_short_week`, `away_short_week`, `home_bye`, `away_bye`) from training. Model A and Model B keep the original 34-feature selection so their published accuracy stays reproducible; the production model uses the corrected, complete feature set since it has no previously-reported number to preserve. See Limitations for details.

**Measured impact of the fix:** after retraining on the corrected 39-feature set, Week 1, 2026 predictions were compared directly against the original 34-feature run. Across all 16 games, probabilities shifted by less than 1 percentage point in 14 of them, with a maximum shift of 2.5 points (HOU vs. BUF) and no game flipping to a different predicted winner. This is a useful, concrete result in its own right: it confirms the 5 missing features had a real but modest effect, consistent with their small coefficient magnitudes (0.001-0.04) in the feature importance chart below, rather than a large, hidden distortion in the original model.

This model is saved separately as `model_a_2026.pkl` / `scaler_a_2026.pkl` and never overwrites `model_a.pkl` or `model_b.pkl`. Use it via:

```bash
python src/predict_week.py --model a_2026 --season 2026 --week 1
```

**Important limitation:** this model has no fresh, comparable holdout accuracy number. All available data went into training it, so there's no untouched test set left to score it against. That's an expected tradeoff for a deployment model, not a flaw, its purpose is generating the best real predictions going forward, not producing a new benchmark figure. The reported 64.4%/64.7% accuracy numbers describe Model A/B's methodology, not this model's real-world performance, which will only be knowable in hindsight once 2026 games are actually played.

`--backtest` is disabled for this model (it would score against 2024-2025, which is in-sample for it and would produce a misleading, inflated number).

---

## Limitations

- **A feature-selection bug excluded 5 features from Model A and Model B's training, found during v2 development.** Both models select training columns via `col.endswith('_diff')`, which silently excludes any column that doesn't follow that naming pattern. `div_game`, `home_short_week`, `away_short_week`, `home_bye`, and `away_bye` are documented Tier 2 features, present in `modeling_table.csv`, but never actually reached training because of this. The reported 64.4%/64.7% accuracy is accurate to what was actually trained (34 features), just not to the full 39-feature set described in Feature Engineering. This is disclosed rather than silently retrained, to keep the published numbers reproducible. The fix (an explicit `FEATURE_COLS` list instead of a naming convention) has been applied to the production model (see Production Model) and will carry into the in-progress Model C; retraining Model A/B with it was deliberately not done, to preserve their published, citable results.
- **A related lookup bug affected live/backtest point-differential features for any week after Week 1 of a season, also found during v2 development.** `predict_week.py`'s point-differential lookup only checked for prior *seasons*, not already-played weeks of the *current* season, so Week 10 predictions would fall back to end-of-prior-season point differential instead of using Weeks 1-9, which had already happened and were more current. This did not affect Model A/B's officially reported 64.4%/64.7% accuracy (computed independently in `train_classifier.py` from full-history features, not through this lookup), but it did affect the separate week-by-week backtest simulation in `predict_week.py --backtest` and `track_season.py`. Fixed to match the same prior-season-or-earlier-this-season pattern already used correctly elsewhere in the file.
- **Hyperparameter selection was not fully held out.** The C-value regularization sweep was evaluated directly on each split's reported test set, rather than on a separate validation period. This likely makes the reported accuracy/log loss numbers slightly optimistic. A stricter setup would hold out a validation season (e.g. 2023) for tuning and report results only on a completely untouched final test set.
- **Statistical significance now tested via McNemar's test.** Comparing Model A and Vegas on the same 284 2025 games (`src/mcnemar_test.py`), McNemar's test with continuity correction gives statistic=0.29, p=0.59. At alpha=0.05, the ~2-point accuracy gap is not statistically significant, this doesn't mean the model and Vegas are proven equivalent, only that this sample size can't confirm the gap is more than random variation. This confirms the "directional, not statistically confirmed" framing this project has used throughout.
- **Vegas comparison has a later information cutoff.** Closing lines incorporate injury reports, weather, and lineup news released much closer to kickoff than this model's frozen pre-week features, so part of the gap reflects a timing difference, not purely a feature gap.
- The model does not incorporate QB starter continuity, injuries, weather, or line movement. These are plausible, untested explanations for most of the remaining gap to Vegas' accuracy, not a verified attribution, isolating how much each missing factor contributes would require adding them individually and re-measuring.
- No individual player performance or advanced player-tracking stats (e.g. NFL Next Gen Stats like CPOE, time to throw, separation) are included, the model works entirely at the team level.
- Rolling stats have thinner history early in a season (or for a franchise's first tracked season in 2016), making early-week predictions noisier than mid/late-season ones.
- Point-in-time features are frozen as of the start of a target week; if used for in-week betting, they wouldn't reflect information released closer to kickoff (final injury reports, weather updates).
- The model was not tested on divisional rivalry upsets or primetime-specific patterns separately, so it may systematically over- or under-perform on those subsets.
- Automated leakage testing currently covers EPA-based rolling stats only, not the full Tier 2 feature set.

---

## Project Structure

```
nfl-game-predictor/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # schedules.csv, team_stats.csv, pbp.csv
│   └── processed/            # team_week_features.csv, modeling_table.csv
├── models/                   # model_a.pkl, scaler_a.pkl, model_b.pkl, scaler_b.pkl,
│                             #   model_a_2026.pkl, scaler_a_2026.pkl (production model)
├── src/
│   ├── data_pipeline.py      # pulls raw data from nflreadpy
│   ├── features.py           # builds the full modeling table
│   ├── feature_utils.py      # shared rolling-window feature helpers
│   ├── train_classifier.py   # trains Model A and Model B, evaluation, Vegas comparison
│   ├── train_production_model.py # trains the no-holdout production model for live 2026 use
│   ├── mcnemar_test.py       # McNemar's test: Model A vs Vegas significance testing
│   ├── predict_week.py       # predicts any season/week via CLI, works for future weeks
│   ├── track_season.py       # season-long accuracy tracker for Model A
│   └── combine_predictions.py # side-by-side Model A vs Model B comparison
├── tests/
│   └── test_features.py      # automated leakage tests
├── reports/                  # feature importance charts, calibration plots
└── outputs/
    └── predictions/          # weekly and backtest prediction CSVs
```

---

## How to Run

```bash
# clone and install
git clone https://github.com/moiz-manzoor/nfl-game-predictor.git
cd nfl-game-predictor
pip install -r requirements.txt
# built and tested on Python 3.11

# pull data and build features
python src/data_pipeline.py
python src/features.py

# train both evaluation models
python src/train_classifier.py

# train the production model (no holdout, for live 2026 predictions)
python src/train_production_model.py

# run McNemar's test: Model A vs Vegas significance testing on the 2025 test set
python src/mcnemar_test.py

# run automated leakage tests
pytest tests/

# backtest across 2024-2025 (Model B: true out-of-sample holdout for both seasons)
python src/predict_week.py --model b --backtest

# predict a specific upcoming week (evaluation model)
python src/predict_week.py --model a --season 2026 --week 1

# predict a specific upcoming week (production model, no holdout)
python src/predict_week.py --model a_2026 --season 2026 --week 1

# track season-to-date accuracy once games have been played
python src/track_season.py

# compare Model A and Model B predictions for the same week
python src/combine_predictions.py
```

---

## Future Work

- **Tier 3 features:** QB starter continuity and CPOE (completion percentage over expectation) from NFL Next Gen Stats, travel distance, dome/outdoor flag, altitude
- **Additional model types:** random forest classifier (same features/target as Model A, to test whether a more flexible model outperforms logistic regression on this feature set) and a linear regression model predicting point differential directly
- **Combined winner + score model:** a single model that predicts both the game winner and the final score together, rather than treating win/loss and point differential as entirely separate tasks
- **Individual offensive player stat projections:** a player-level model predicting passing attempts, completions, yards, and touchdowns; rushing attempts, yards, and touchdowns; and targets, receptions, receiving yards, and receiving touchdowns, extending the project from team-level outcomes down to individual player performance
- **Strength-of-schedule adjustment** for EPA and success rate
- Automated weekly runs as the season progresses

---

## Author

Moiz Manzoor — [GitHub](https://github.com/moiz-manzoor) · [LinkedIn](https://linkedin.com/in/moiz-m)
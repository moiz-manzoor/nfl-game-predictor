"""
track_season.py

Season-long accuracy tracker for Model A (the primary model). Checks
schedules.csv for which weeks of the target season have actually been
played (home_score not null), re-runs predict_week() for each of those
completed weeks, and saves a cumulative accuracy record comparing Model A
against Vegas.

Overwrites the tracker CSV fresh each run with the full season-to-date, so
it's always a complete, up-to-date record rather than an incremental log.
Run this after each week's games finish, following a fresh data_pipeline.py
pull, to see how every prior week has performed.
"""

import pandas as pd
import joblib
import os
from predict_week import predict_week, pbp, schedules

SEASON = 2026
MODEL_LETTER = 'a'

model = joblib.load(f'models/model_{MODEL_LETTER}.pkl')
scaler = joblib.load(f'models/scaler_{MODEL_LETTER}.pkl')

# --- Find which weeks of this season have actually been played ---
season_games = schedules[schedules['season'] == SEASON]
played_weeks = sorted(
    season_games[season_games['home_score'].notna()]['week'].unique().tolist()
)

if not played_weeks:
    print(f"No completed games found yet for {SEASON}. Nothing to track.")
else:
    print(f"Found completed weeks for {SEASON}: {played_weeks}")

    all_results = []
    for wk in played_weeks:
        result = predict_week(SEASON, wk, pbp, model, scaler)
        if result is not None:
            all_results.append(result)
            acc = result['correct'].mean() * 100
            vegas_acc = result['vegas_correct'].mean() * 100
            print(f"{SEASON} Week {wk} — Model A: {acc:.1f}% | Vegas: {vegas_acc:.1f}%")

    combined = pd.concat(all_results, ignore_index=True)

    output_cols = ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score',
                   'home_win_prob', 'predicted_winner', 'actual_winner', 'correct',
                   'home_implied_prob', 'vegas_correct']
    combined_output = combined[output_cols]

    os.makedirs("outputs/predictions", exist_ok=True)
    out_path = f"outputs/predictions/season{SEASON}_modela_tracker.csv"
    combined_output.to_csv(out_path, index=False)

    print(f"\nSeason-to-date Model A accuracy: {combined['correct'].mean()*100:.1f}%")
    print(f"Season-to-date Vegas accuracy: {combined['vegas_correct'].mean()*100:.1f}%")
    print(f"Saved {out_path} ({len(combined)} games across {len(played_weeks)} weeks)")
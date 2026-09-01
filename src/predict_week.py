"""
predict_week.py

Generates predictions for a specific NFL season/week using a trained model
(Model A or Model B from train_classifier.py). Rebuilds the same features
as features.py, but scoped to only the data available as of the target
week, so this works equally well for:
- Backtesting a past week (real outcomes available, accuracy is computed)
- Predicting a genuinely future week (no outcomes yet, prediction only)

Usage:
    python src/predict_week.py --model a --season 2026 --week 1
    python src/predict_week.py --model b --backtest   (2024-2025 backtest)
    python src/predict_week.py --model a_2026 --season 2026 --week 1
        (production model, trained on 2016-2025 with no holdout, for live 2026 games)

Key leakage-prevention detail: latest_epa and latest_point_diff both grab
each team's single most recent row by team only (not team+season). This
matters for Week 1 of a new season, where there's no in-season history yet
to roll from, the team's most recent game (last week of the prior season)
is used instead.
"""

import pandas as pd
import joblib
import os
import argparse
from features import modeling_cols, schedules
from feature_utils import apply_rolling_windows, build_sack_rate, build_pace, build_third_down_rate, build_explosive_rate, build_red_zone_rate, build_turnover_margin


def moneyline_to_prob(moneyline):
    """Convert an American moneyline to a raw (un-de-vigged) implied probability."""
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)


pbp = pd.read_csv("data/raw/pbp.csv", low_memory=False)

# Build point_diff (rolling scoring margin) for each team, once
home_rows = schedules[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
home_rows.columns = ['season', 'week', 'team', 'opponent', 'points_for', 'points_allowed']

away_rows = schedules[['season', 'week', 'away_team', 'home_team', 'away_score', 'home_score']].copy()
away_rows.columns = ['season', 'week', 'team', 'opponent', 'points_for', 'points_allowed']

team_games = pd.concat([home_rows, away_rows], ignore_index=True)

team_name_fixes = {'OAK': 'LV', 'SD': 'LAC'}
team_games['team'] = team_games['team'].replace(team_name_fixes)
team_games['opponent'] = team_games['opponent'].replace(team_name_fixes)

team_games['point_diff'] = team_games['points_for'] - team_games['points_allowed']
team_games = team_games.sort_values(['team', 'season', 'week'])

team_games['point_diff_last3'] = team_games.groupby('team')['point_diff'].transform(lambda x: x.shift(1).rolling(3).mean())
team_games['point_diff_last5'] = team_games.groupby('team')['point_diff'].transform(lambda x: x.shift(1).rolling(5).mean())
team_games['point_diff_s2d'] = team_games.groupby(['team', 'season'])['point_diff'].transform(lambda x: x.shift(1).expanding().mean())


def predict_week(season, week, pbp, model, scaler):
    """
    Build features and generate predictions for one season/week.

    Returns a DataFrame with one row per game, including home_win_prob,
    predicted_winner, and (if the game has already been played) accuracy
    columns comparing the model and Vegas against the actual result.
    Returns None if the target season/week isn't found in schedules.
    """
    # Each team's most recent rolling point-diff stats, grouped by team only
    # (not team+season) so Week 1 of a new season can still pull from the
    # end of the prior season.
    latest_point_diff = (
        team_games[team_games['season'] < season]
        .sort_values(['team', 'season', 'week'])
        .groupby('team', as_index=False)
        .tail(1)[['team', 'point_diff_last3', 'point_diff_last5', 'point_diff_s2d']]
    )

    pbp_prior = pbp[
        (pbp['season'] < season) |
        ((pbp['season'] == season) & (pbp['week'] < week))
    ].copy()

    pbp_filtered = pbp_prior[pbp_prior['play_type'].isin(['pass', 'run'])]
    pbp_filtered = pbp_filtered.copy()

    sack_stats = build_sack_rate(pbp_filtered)
    pace_stats = build_pace(pbp_filtered)

    pbp_filtered['turnover'] = pbp_filtered['interception'] + pbp_filtered['fumble_lost']
    turnovers_committed = pbp_filtered.groupby(['posteam', 'season', 'week'])['turnover'].sum().reset_index()
    turnovers_committed = turnovers_committed.rename(columns={'posteam': 'team', 'turnover': 'turnovers_committed'})

    turnovers_forced = pbp_filtered.groupby(['defteam', 'season', 'week'])['turnover'].sum().reset_index()
    turnovers_forced = turnovers_forced.rename(columns={'defteam': 'team', 'turnover': 'turnovers_forced'})

    turnovers = build_turnover_margin(turnovers_committed, turnovers_forced)

    pbp_filtered['third_down_attempt'] = pbp_filtered['third_down_converted'] + pbp_filtered['third_down_failed']
    third_down = pbp_filtered.groupby(['posteam', 'season', 'week']).agg(
        conversions=('third_down_converted', 'sum'),
        attempts=('third_down_attempt', 'sum')
    ).reset_index()
    third_down = build_third_down_rate(third_down)

    pbp_filtered['explosive'] = (
        ((pbp_filtered['play_type'] == 'run') & (pbp_filtered['yards_gained'] >= 10)) |
        ((pbp_filtered['play_type'] == 'pass') & (pbp_filtered['yards_gained'] >= 15))
    ).astype(int)

    explosive_plays = pbp_filtered.groupby(['posteam', 'season', 'week']).agg(
        explosive_plays=('explosive', 'sum'),
        total_plays=('explosive', 'count')
    ).reset_index()
    explosive_plays = build_explosive_rate(explosive_plays)

    drives = pbp_prior.drop_duplicates(subset=['posteam', 'season', 'week', 'fixed_drive'])[
        ['posteam', 'season', 'week', 'fixed_drive', 'drive_inside20', 'fixed_drive_result']
    ]
    drives = drives.dropna(subset=['drive_inside20'])

    red_zone_drives = drives[drives['drive_inside20'] == 1].copy()
    red_zone_drives['red_zone_td'] = (red_zone_drives['fixed_drive_result'] == 'Touchdown').astype(int)

    red_zone = red_zone_drives.groupby(['posteam', 'season', 'week']).agg(
        red_zone_tds=('red_zone_td', 'sum'),
        red_zone_trips=('red_zone_td', 'count')
    ).reset_index()
    red_zone = build_red_zone_rate(red_zone)

    off_epa = pbp_filtered.groupby(['posteam', 'season', 'week'])['epa'].mean().reset_index()
    def_epa = pbp_filtered.groupby(['defteam', 'season', 'week'])['epa'].mean().reset_index()
    off_success = pbp_filtered.groupby(['posteam', 'season', 'week'])['success'].mean().reset_index()
    def_success = pbp_filtered.groupby(['defteam', 'season', 'week'])['success'].mean().reset_index()

    team_week_epa = off_epa.merge(
        def_epa,
        left_on=['posteam', 'season', 'week'],
        right_on=['defteam', 'season', 'week'],
        how='left'
    )
    team_week_epa = team_week_epa.merge(
        off_success,
        left_on=['posteam', 'season', 'week'],
        right_on=['posteam', 'season', 'week'],
        how='left'
    )
    team_week_epa = team_week_epa.merge(
        def_success,
        left_on=['posteam', 'season', 'week'],
        right_on=['defteam', 'season', 'week'],
        how='left'
    )

    team_week_epa = team_week_epa.rename(columns={
        'posteam': 'team',
        'epa_x': 'off_epa',
        'epa_y': 'def_epa',
        'success_x': 'off_success',
        'success_y': 'def_success'
    })
    team_week_epa = team_week_epa.drop(columns=['defteam_x', 'defteam_y'])
    team_week_epa = team_week_epa.sort_values(['team', 'season', 'week'])

    team_week_epa = apply_rolling_windows(team_week_epa, 'off_epa')
    team_week_epa = apply_rolling_windows(team_week_epa, 'def_epa')
    team_week_epa = apply_rolling_windows(team_week_epa, 'off_success')
    team_week_epa = apply_rolling_windows(team_week_epa, 'def_success')

    team_week_epa = team_week_epa.merge(
        turnovers[['team', 'season', 'week', 'turnover_margin', 'turnover_margin_last3', 'turnover_margin_last5', 'turnover_margin_s2d']],
        on=['team', 'season', 'week'],
        how='left'
    )
    team_week_epa = team_week_epa.merge(
        third_down[['team', 'season', 'week', 'third_down_rate', 'third_down_rate_last3', 'third_down_rate_last5', 'third_down_rate_s2d']],
        on=['team', 'season', 'week'],
        how='left'
    )
    team_week_epa = team_week_epa.merge(
        sack_stats[['team', 'week', 'season', 'sack_rate', 'sack_rate_last3', 'sack_rate_last5', 'sack_rate_s2d']],
        on=['team', 'week', 'season'],
        how='left'
    )
    team_week_epa = team_week_epa.merge(
        explosive_plays[['team', 'season', 'week', 'explosive_rate', 'explosive_rate_last3', 'explosive_rate_last5', 'explosive_rate_s2d']],
        on=['team', 'season', 'week'],
        how='left'
    )
    team_week_epa = team_week_epa.merge(
        red_zone[['team', 'season', 'week', 'red_zone_td_rate', 'red_zone_td_rate_last3', 'red_zone_td_rate_last5', 'red_zone_td_rate_s2d']],
        on=['team', 'season', 'week'],
        how='left'
    )
    team_week_epa = team_week_epa.merge(
        pace_stats[['team', 'week', 'season', 'plays_per_game', 'plays_per_game_last3', 'plays_per_game_last5', 'plays_per_game_s2d']],
        on=['team', 'week', 'season'],
        how='left'
    )

    target_games = schedules[(schedules['season'] == season) & (schedules['week'] == week)][
        ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score',
         'home_rest', 'away_rest', 'div_game', 'home_moneyline', 'away_moneyline']
    ].copy()

    if target_games.empty:
        return None

    # Same team-only grouping as latest_point_diff above, for the same reason.
    latest_epa = (
        team_week_epa
        .sort_values(['team', 'season', 'week'])
        .groupby('team', as_index=False)
        .tail(1)
    )

    home = latest_epa.add_prefix('home_')
    away = latest_epa.add_prefix('away_')

    games = target_games.merge(
        home,
        left_on=['home_team'],
        right_on=['home_team'],
        how='left'
    )
    games = games.merge(
        away,
        left_on=['away_team'],
        right_on=['away_team'],
        how='left'
    )

    games['home_short_week'] = (games['home_rest'] <= 4).astype(int)
    games['away_short_week'] = (games['away_rest'] <= 4).astype(int)
    games['home_bye'] = (games['home_rest'] >= 10).astype(int)
    games['away_bye'] = (games['away_rest'] >= 10).astype(int)
    games['rest_diff'] = games['home_rest'] - games['away_rest']

    games = games.merge(
        latest_point_diff.add_prefix('home_'),
        left_on=['home_team'],
        right_on=['home_team'],
        how='left'
    )
    games = games.merge(
        latest_point_diff.add_prefix('away_'),
        left_on=['away_team'],
        right_on=['away_team'],
        how='left'
    )

    diffs = {
        'off_epa_last3_diff': games['home_off_epa_last3'] - games['away_off_epa_last3'],
        'off_epa_last5_diff': games['home_off_epa_last5'] - games['away_off_epa_last5'],
        'off_epa_s2d_diff': games['home_off_epa_s2d'] - games['away_off_epa_s2d'],

        'def_epa_last3_diff': games['home_def_epa_last3'] - games['away_def_epa_last3'],
        'def_epa_last5_diff': games['home_def_epa_last5'] - games['away_def_epa_last5'],
        'def_epa_s2d_diff': games['home_def_epa_s2d'] - games['away_def_epa_s2d'],

        'off_success_last3_diff': games['home_off_success_last3'] - games['away_off_success_last3'],
        'off_success_last5_diff': games['home_off_success_last5'] - games['away_off_success_last5'],
        'off_success_s2d_diff': games['home_off_success_s2d'] - games['away_off_success_s2d'],

        'def_success_last3_diff': games['home_def_success_last3'] - games['away_def_success_last3'],
        'def_success_last5_diff': games['home_def_success_last5'] - games['away_def_success_last5'],
        'def_success_s2d_diff': games['home_def_success_s2d'] - games['away_def_success_s2d'],

        'turnover_margin_last3_diff': games['home_turnover_margin_last3'] - games['away_turnover_margin_last3'],
        'turnover_margin_last5_diff': games['home_turnover_margin_last5'] - games['away_turnover_margin_last5'],
        'turnover_margin_s2d_diff': games['home_turnover_margin_s2d'] - games['away_turnover_margin_s2d'],

        'third_down_rate_last3_diff': games['home_third_down_rate_last3'] - games['away_third_down_rate_last3'],
        'third_down_rate_last5_diff': games['home_third_down_rate_last5'] - games['away_third_down_rate_last5'],
        'third_down_rate_s2d_diff': games['home_third_down_rate_s2d'] - games['away_third_down_rate_s2d'],

        'sack_rate_last3_diff': games['home_sack_rate_last3'] - games['away_sack_rate_last3'],
        'sack_rate_last5_diff': games['home_sack_rate_last5'] - games['away_sack_rate_last5'],
        'sack_rate_s2d_diff': games['home_sack_rate_s2d'] - games['away_sack_rate_s2d'],

        'explosive_rate_last3_diff': games['home_explosive_rate_last3'] - games['away_explosive_rate_last3'],
        'explosive_rate_last5_diff': games['home_explosive_rate_last5'] - games['away_explosive_rate_last5'],
        'explosive_rate_s2d_diff': games['home_explosive_rate_s2d'] - games['away_explosive_rate_s2d'],

        'red_zone_td_rate_last3_diff': games['home_red_zone_td_rate_last3'] - games['away_red_zone_td_rate_last3'],
        'red_zone_td_rate_last5_diff': games['home_red_zone_td_rate_last5'] - games['away_red_zone_td_rate_last5'],
        'red_zone_td_rate_s2d_diff': games['home_red_zone_td_rate_s2d'] - games['away_red_zone_td_rate_s2d'],

        'plays_per_game_last3_diff': games['home_plays_per_game_last3'] - games['away_plays_per_game_last3'],
        'plays_per_game_last5_diff': games['home_plays_per_game_last5'] - games['away_plays_per_game_last5'],
        'plays_per_game_s2d_diff': games['home_plays_per_game_s2d'] - games['away_plays_per_game_s2d'],

        'point_diff_last3_diff': games['home_point_diff_last3'] - games['away_point_diff_last3'],
        'point_diff_last5_diff': games['home_point_diff_last5'] - games['away_point_diff_last5'],
        'point_diff_s2d_diff': games['home_point_diff_s2d'] - games['away_point_diff_s2d'],
    }

    games = pd.concat([games, pd.DataFrame(diffs, index=games.index)], axis=1)

    predict_cols = [c for c in modeling_cols if c != 'home_team_win']
    week_features = games[
        ['season', 'week', 'home_team', 'away_team'] +
        [c for c in predict_cols if c not in ['home_season', 'home_week', 'home_team', 'away_team']]
    ].copy()
    week_features = week_features.fillna(0)

    predict_cols_final = [c for c in predict_cols if c.endswith('_diff')]
    X = week_features[predict_cols_final]
    X_scaled = scaler.transform(X)

    week_features['home_win_prob'] = model.predict_proba(X_scaled)[:, 1]
    week_features['predicted_winner'] = week_features.apply(
        lambda row: row['home_team'] if row['home_win_prob'] >= 0.5 else row['away_team'],
        axis=1
    )

    week_features['home_score'] = games['home_score']
    week_features['away_score'] = games['away_score']
    week_features['actual_winner'] = week_features.apply(
        lambda row: row['home_team'] if row['home_score'] > row['away_score'] else row['away_team'],
        axis=1
    )
    week_features['correct'] = week_features['predicted_winner'] == week_features['actual_winner']

    week_odds = games[['home_team', 'away_team', 'home_moneyline', 'away_moneyline']].copy()
    week_odds['home_implied_prob_raw'] = week_odds['home_moneyline'].apply(moneyline_to_prob)
    week_odds['away_implied_prob_raw'] = week_odds['away_moneyline'].apply(moneyline_to_prob)
    week_odds['vig_total'] = week_odds['home_implied_prob_raw'] + week_odds['away_implied_prob_raw']
    week_odds['home_implied_prob'] = week_odds['home_implied_prob_raw'] / week_odds['vig_total']

    week_features = week_features.merge(
        week_odds[['home_team', 'away_team', 'home_implied_prob']],
        on=['home_team', 'away_team'],
        how='left'
    )
    week_features['vegas_pred'] = (week_features['home_implied_prob'] > 0.5).astype(int)
    week_features['vegas_correct'] = (
        ((week_features['vegas_pred'] == 1) & (week_features['home_team'] == week_features['actual_winner'])) |
        ((week_features['vegas_pred'] == 0) & (week_features['away_team'] == week_features['actual_winner']))
    )

    return week_features


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Predict NFL games with Model A or Model B")
    parser.add_argument('--model', type=str, choices=['a', 'b', 'a_2026'], default='a',
                         help="Which trained model to use: 'a' (test=2025), 'b' (test=2024-2025), "
                              "or 'a_2026' (production model, trained on 2016-2025 with no holdout, for live 2026 predictions). Default: a")
    parser.add_argument('--season', type=int, help="Season to predict, e.g. 2026")
    parser.add_argument('--week', type=int, help="Week number to predict, e.g. 2")
    parser.add_argument('--backtest', action='store_true',
                         help="Run the full 2024-2025 backtest and print week-by-week accuracy")
    args = parser.parse_args()

    # Load the chosen model
    model_label = args.model.upper()
    model = joblib.load(f'models/model_{args.model}.pkl')
    scaler = joblib.load(f'models/scaler_{args.model}.pkl')
    print(f"Loaded Model {model_label}")

    # Optional: run across multiple weeks and aggregate
    if args.backtest and args.model == 'a_2026':
        print("--backtest is not meaningful for the production model (a_2026): "
              "2024-2025 is in-sample for it, not a real holdout. "
              "Use --model a or --model b for backtesting instead.")
        args.backtest = False

    if args.backtest:
        all_results = []

        for season in [2024, 2025]:
            for wk in range(1, 18):
                result = predict_week(season, wk, pbp, model, scaler)
                if result is not None:
                    all_results.append(result)
                    print(f"{season} Week {wk} done — {result['correct'].mean()*100:.1f}% model, {result['vegas_correct'].mean()*100:.1f}% Vegas")

        combined = pd.concat(all_results, ignore_index=True)

        output_cols = ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score',
                       'home_win_prob', 'predicted_winner', 'actual_winner', 'correct',
                       'home_implied_prob', 'vegas_correct']
        combined_output = combined[output_cols]

        os.makedirs("outputs/predictions", exist_ok=True)
        combined_output.to_csv(f"outputs/predictions/model_{args.model}_2024_2025_predictions.csv", index=False)

        print(f"\nOverall Model {model_label} accuracy: {combined['correct'].mean()*100:.1f}%")
        print(f"Overall Vegas accuracy: {combined['vegas_correct'].mean()*100:.1f}%")
        print(f"Saved model_{args.model}_2024_2025_predictions.csv ({len(combined)} games)")

    # --- Predict a single real, upcoming week ---
    if args.season is not None and args.week is not None:
        target_season = args.season
        target_week = args.week

        week_result = predict_week(target_season, target_week, pbp, model, scaler)

        if week_result is not None:
            print(f"\n{target_season} Week {target_week} predictions (Model {model_label}):")
            print(week_result[['home_team', 'away_team', 'home_win_prob',
                                'predicted_winner', 'home_implied_prob']])

            os.makedirs("outputs/predictions", exist_ok=True)
            out_path = f"outputs/predictions/week{target_week}_{target_season}_model{args.model}_predictions.csv"
            week_result.to_csv(out_path, index=False)
            print(f"Saved {out_path}")
        else:
            print(f"predict_week returned None for {target_season} Week {target_week}, check that the schedule has this week")
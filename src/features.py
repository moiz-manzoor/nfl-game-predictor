"""
features.py

Builds the full game-level modeling table used to train the classifiers.

Pipeline:
1. Load raw schedules + play-by-play data
2. Build team-week level stats (EPA, success rate, turnovers, third down,
   sacks, explosive plays, red zone, pace, point differential), each with
   leakage-safe rolling windows (last3/last5/season-to-date) via feature_utils
3. Assemble one row per game by joining home/away team-week stats
4. Compute home-minus-away differentials for every stat
5. Save the final modeling table to data/processed/modeling_table.csv

Leakage rule: modeling_cols only includes the rolling (_last3/_last5/_s2d)
differential columns, never the raw same-game differentials (e.g.
off_epa_diff). Raw differentials reflect the outcome of the game being
predicted and would leak the target into the features.
"""

import pandas as pd
from feature_utils import apply_rolling_windows, build_sack_rate, build_pace, build_turnover_margin, build_third_down_rate, build_explosive_rate, build_red_zone_rate

pbp = pd.read_csv("data/raw/pbp.csv", low_memory=False)
schedules = pd.read_csv("data/raw/schedules.csv")

# --- Build point_diff (rolling scoring margin) for each team ---
home_rows = schedules[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'home_rest', 'div_game']].copy()
home_rows.columns = ['season', 'week', 'team', 'opponent', 'points_for', 'points_allowed', 'rest', 'div_game']
home_rows['is_home'] = 1
away_rows = schedules[['season', 'week', 'away_team', 'home_team', 'away_score', 'home_score', 'away_rest', 'div_game']].copy()
away_rows.columns = ['season', 'week', 'team', 'opponent', 'points_for', 'points_allowed', 'rest', 'div_game']
away_rows['is_home'] = 0

team_games = pd.concat([home_rows, away_rows], ignore_index=True)

# Fix franchise relocation naming mismatches between pbp and schedules data
team_name_fixes = {'OAK': 'LV', 'SD': 'LAC'}
team_games['team'] = team_games['team'].replace(team_name_fixes)
team_games['opponent'] = team_games['opponent'].replace(team_name_fixes)
team_games['point_diff'] = team_games['points_for'] - team_games['points_allowed']
team_games = team_games.sort_values(['team', 'season', 'week'])
team_games['point_diff_last5'] = team_games.groupby('team')['point_diff'].transform(lambda x: x.shift(1).rolling(5).mean())
team_games['point_diff_last3'] = team_games.groupby('team')['point_diff'].transform(lambda x: x.shift(1).rolling(3).mean())
team_games['point_diff_s2d'] = team_games.groupby(['team', 'season'])['point_diff'].transform(lambda x: x.shift(1).expanding().mean())


# --- Filter play-by-play to plays that count as offensive/defensive snaps ---
# Excludes kneels, spikes, timeouts, etc. that aren't meaningful plays
pbp_filtered = pbp[pbp['play_type'].isin(['pass', 'run'])]
pbp_filtered = pbp_filtered.copy()


# --- Turnover margin ---
pbp_filtered['turnover'] = pbp_filtered['interception'] + pbp_filtered['fumble_lost']
turnovers_committed = pbp_filtered.groupby(['posteam', 'season', 'week'])['turnover'].sum().reset_index()
turnovers_committed = turnovers_committed.rename(columns={'posteam': 'team', 'turnover': 'turnovers_committed'})

turnovers_forced = pbp_filtered.groupby(['defteam', 'season', 'week'])['turnover'].sum().reset_index()
turnovers_forced = turnovers_forced.rename(columns={'defteam': 'team', 'turnover': 'turnovers_forced'})

turnovers = build_turnover_margin(turnovers_committed, turnovers_forced)


# --- Third down conversion rate ---
pbp_filtered['third_down_attempt'] = pbp_filtered['third_down_converted'] + pbp_filtered['third_down_failed']
third_down = pbp_filtered.groupby(['posteam', 'season', 'week']).agg(
    conversions=('third_down_converted', 'sum'),
    attempts=('third_down_attempt', 'sum')
).reset_index()
third_down = build_third_down_rate(third_down)

# --- Sack rate ---
sack_stats = build_sack_rate(pbp_filtered)


# --- Explosive play rate (PFF thresholds: 10+ yd runs, 15+ yd passes) ---
pbp_filtered['explosive'] = (
    ((pbp_filtered['play_type'] == 'run') & (pbp_filtered['yards_gained'] >= 10)) |
    ((pbp_filtered['play_type'] == 'pass') & (pbp_filtered['yards_gained'] >= 15))
).astype(int)

explosive_plays = pbp_filtered.groupby(['posteam', 'season', 'week']).agg(
    explosive_plays=('explosive', 'sum'),
    total_plays=('explosive', 'count')
).reset_index()
explosive_plays = build_explosive_rate(explosive_plays)

# --- Red zone touchdown rate ---
# Red zone = plays of a drive that start inside the opponent's 20-yard line.
# Uses unfiltered pbp (not pbp_filtered) since drive-ending plays may not be pass/run plays.
drives = pbp.drop_duplicates(subset=['posteam', 'season', 'week', 'fixed_drive'])[
    ['posteam', 'season', 'week', 'fixed_drive', 'drive_inside20', 'fixed_drive_result']
]
drives = drives.dropna(subset=['posteam', 'drive_inside20'])

red_zone_drives = drives[drives['drive_inside20'] == 1].copy()
red_zone_drives['red_zone_td'] = (red_zone_drives['fixed_drive_result'] == 'Touchdown').astype(int)

red_zone = red_zone_drives.groupby(['posteam', 'season', 'week']).agg(
    red_zone_tds=('red_zone_td', 'sum'),
    red_zone_trips=('red_zone_td', 'count')
).reset_index()
red_zone = build_red_zone_rate(red_zone)


# --- Pace (plays per game) ---
pace_stats = build_pace(pbp_filtered)


# --- Offensive/defensive EPA and success rate ---
off_epa = pbp_filtered.groupby(['posteam', 'season', 'week'])['epa'].mean().reset_index()
def_epa = pbp_filtered.groupby(['defteam', 'season', 'week'])['epa'].mean().reset_index()

off_success = pbp_filtered.groupby(['posteam', 'season', 'week'])['success'].mean().reset_index()
def_success = pbp_filtered.groupby(['defteam', 'season', 'week'])['success'].mean().reset_index()

# Merge offensive and defensive EPA/success into one team-week table
team_week_epa = off_epa.merge(
    def_epa,
    left_on=['posteam', 'season', 'week'],
    right_on=['defteam', 'season', 'week']
)

team_week_epa = team_week_epa.merge(
    off_success,
    left_on=['posteam', 'season', 'week'],
    right_on=['posteam', 'season', 'week']
)

team_week_epa = team_week_epa.merge(
    def_success,
    left_on=['posteam', 'season', 'week'],
    right_on=['defteam', 'season', 'week']
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

# Rolling EPA/success features, shifted by 1 to prevent leakage. Each row's
# rolling average only includes strictly prior games, never the game being
# predicted. last3/last5 use fixed windows; s2d resets each season.
team_week_epa['off_epa_last3'] = team_week_epa.groupby('team')['off_epa'].transform(lambda x: x.shift(1).rolling(3).mean())
team_week_epa['def_epa_last3'] = team_week_epa.groupby('team')['def_epa'].transform(lambda x: x.shift(1).rolling(3).mean())
team_week_epa['off_epa_last5'] = team_week_epa.groupby('team')['off_epa'].transform(lambda x: x.shift(1).rolling(5).mean())
team_week_epa['def_epa_last5'] = team_week_epa.groupby('team')['def_epa'].transform(lambda x: x.shift(1).rolling(5).mean())
team_week_epa['off_epa_s2d'] = team_week_epa.groupby(['team', 'season'])['off_epa'].transform(lambda x: x.shift(1).expanding().mean())
team_week_epa['def_epa_s2d'] = team_week_epa.groupby(['team', 'season'])['def_epa'].transform(lambda x: x.shift(1).expanding().mean())
team_week_epa['off_success_last3'] = team_week_epa.groupby('team')['off_success'].transform(lambda x: x.shift(1).rolling(3).mean())
team_week_epa['def_success_last3'] = team_week_epa.groupby('team')['def_success'].transform(lambda x: x.shift(1).rolling(3).mean())
team_week_epa['off_success_last5'] = team_week_epa.groupby('team')['off_success'].transform(lambda x: x.shift(1).rolling(5).mean())
team_week_epa['def_success_last5'] = team_week_epa.groupby('team')['def_success'].transform(lambda x: x.shift(1).rolling(5).mean())
team_week_epa['off_success_s2d'] = team_week_epa.groupby(['team', 'season'])['off_success'].transform(lambda x: x.shift(1).expanding().mean())
team_week_epa['def_success_s2d'] = team_week_epa.groupby(['team', 'season'])['def_success'].transform(lambda x: x.shift(1).expanding().mean())


# Merge turnover margin (raw + rolling) into team_week_epa
team_week_epa = team_week_epa.merge(
    turnovers[['team', 'season', 'week', 'turnover_margin', 'turnover_margin_last3', 'turnover_margin_last5', 'turnover_margin_s2d']],
    on=['team', 'season', 'week'],
    how='left'
)

# Merge third down rate (raw + rolling) into team_week_epa
team_week_epa = team_week_epa.merge(
    third_down[['team', 'season', 'week', 'third_down_rate', 'third_down_rate_last3', 'third_down_rate_last5', 'third_down_rate_s2d']],
    on=['team', 'season', 'week'],
    how='left'
)

# Merge sack rate (raw + rolling) into team_week_epa
team_week_epa = team_week_epa.merge(
    sack_stats[['team', 'week', 'season', 'sack_rate', 'sack_rate_last3', 'sack_rate_last5', 'sack_rate_s2d']],
    on=['team', 'week', 'season'],
    how='left'
)

# Merge explosive play rate (raw + rolling) into team_week_epa
team_week_epa = team_week_epa.merge(
    explosive_plays[['team', 'season', 'week', 'explosive_rate', 'explosive_rate_last3', 'explosive_rate_last5', 'explosive_rate_s2d']],
    on=['team', 'season', 'week'],
    how='left'
)

# Merge red zone touchdown rate (raw + rolling) into team_week_epa
team_week_epa = team_week_epa.merge(
    red_zone[['team', 'season', 'week', 'red_zone_td_rate', 'red_zone_td_rate_last3', 'red_zone_td_rate_last5', 'red_zone_td_rate_s2d']],
    on=['team', 'season', 'week'],
    how='left'
)

# Merge pace (raw + rolling) into team_week_epa
team_week_epa = team_week_epa.merge(
    pace_stats[['team', 'week', 'season', 'plays_per_game', 'plays_per_game_last3', 'plays_per_game_last5', 'plays_per_game_s2d']],
    on=['team', 'week', 'season'],
    how='left'
)

# Reorder columns for readability (no effect on modeling)
team_week_epa = team_week_epa[[
    'team', 'season', 'week',
    'off_epa', 'off_epa_last3', 'off_epa_last5', 'off_epa_s2d',
    'def_epa', 'def_epa_last3', 'def_epa_last5', 'def_epa_s2d',
    'off_success', 'off_success_last3', 'off_success_last5', 'off_success_s2d',
    'def_success', 'def_success_last3', 'def_success_last5', 'def_success_s2d',
    'turnover_margin', 'turnover_margin_last3', 'turnover_margin_last5', 'turnover_margin_s2d',
    'third_down_rate', 'third_down_rate_last3', 'third_down_rate_last5', 'third_down_rate_s2d',
    'sack_rate', 'sack_rate_last3', 'sack_rate_last5', 'sack_rate_s2d',
    'explosive_rate', 'explosive_rate_last3', 'explosive_rate_last5', 'explosive_rate_s2d',
    'red_zone_td_rate', 'red_zone_td_rate_last3', 'red_zone_td_rate_last5', 'red_zone_td_rate_s2d',
    'plays_per_game', 'plays_per_game_last3', 'plays_per_game_last5', 'plays_per_game_s2d'
]]

final_features = team_week_epa.merge(team_games, on=['team', 'season', 'week'])
final_features.to_csv("data/processed/team_week_features.csv", index=False)

# --- Split team-week rows into home/away perspective, prefix columns, and merge into one row per game ---
home = final_features[final_features['is_home'] == 1].copy()
away = final_features[final_features['is_home'] == 0].copy()

home = home.add_prefix('home_')
away = away.add_prefix('away_')

games = home.merge(
    away,
    left_on=['home_season', 'home_week', 'home_team', 'home_opponent'],
    right_on=['away_season', 'away_week', 'away_opponent', 'away_team']
)

games['div_game'] = games['home_div_game']

games['home_short_week'] = (games['home_rest'] <= 4).astype(int)
games['away_short_week'] = (games['away_rest'] <= 4).astype(int)
games['home_bye'] = (games['home_rest'] >= 10).astype(int)
games['away_bye'] = (games['away_rest'] >= 10).astype(int)

# --- Home-minus-away differentials for every feature ---
# Note: raw (non-rolling) differentials like off_epa_diff are computed here
# for completeness but are deliberately excluded from modeling_cols below,
# since they reflect the outcome of the game being predicted (leakage).

# EPA differentials
games['off_epa_diff'] = games['home_off_epa'] - games['away_off_epa']
games['off_epa_last3_diff'] = games['home_off_epa_last3'] - games['away_off_epa_last3']
games['off_epa_last5_diff'] = games['home_off_epa_last5'] - games['away_off_epa_last5']
games['off_epa_s2d_diff'] = games['home_off_epa_s2d'] - games['away_off_epa_s2d']

games['def_epa_diff'] = games['home_def_epa'] - games['away_def_epa']
games['def_epa_last3_diff'] = games['home_def_epa_last3'] - games['away_def_epa_last3']
games['def_epa_last5_diff'] = games['home_def_epa_last5'] - games['away_def_epa_last5']
games['def_epa_s2d_diff'] = games['home_def_epa_s2d'] - games['away_def_epa_s2d']

# Success rate differentials
games['off_success_diff'] = games['home_off_success'] - games['away_off_success']
games['off_success_last3_diff'] = games['home_off_success_last3'] - games['away_off_success_last3']
games['off_success_last5_diff'] = games['home_off_success_last5'] - games['away_off_success_last5']
games['off_success_s2d_diff'] = games['home_off_success_s2d'] - games['away_off_success_s2d']

games['def_success_diff'] = games['home_def_success'] - games['away_def_success']
games['def_success_last3_diff'] = games['home_def_success_last3'] - games['away_def_success_last3']
games['def_success_last5_diff'] = games['home_def_success_last5'] - games['away_def_success_last5']
games['def_success_s2d_diff'] = games['home_def_success_s2d'] - games['away_def_success_s2d']

# Point differential differentials
games['point_diff_diff'] = games['home_point_diff'] - games['away_point_diff']
games['point_diff_last3_diff'] = games['home_point_diff_last3'] - games['away_point_diff_last3']
games['point_diff_last5_diff'] = games['home_point_diff_last5'] - games['away_point_diff_last5']
games['point_diff_s2d_diff'] = games['home_point_diff_s2d'] - games['away_point_diff_s2d']

# Turnover margin differentials
games['turnover_margin_diff'] = games['home_turnover_margin'] - games['away_turnover_margin']
games['turnover_margin_last3_diff'] = games['home_turnover_margin_last3'] - games['away_turnover_margin_last3']
games['turnover_margin_last5_diff'] = games['home_turnover_margin_last5'] - games['away_turnover_margin_last5']
games['turnover_margin_s2d_diff'] = games['home_turnover_margin_s2d'] - games['away_turnover_margin_s2d']

# Third down rate differentials
games['third_down_rate_last3_diff'] = games['home_third_down_rate_last3'] - games['away_third_down_rate_last3']
games['third_down_rate_last5_diff'] = games['home_third_down_rate_last5'] - games['away_third_down_rate_last5']
games['third_down_rate_s2d_diff'] = games['home_third_down_rate_s2d'] - games['away_third_down_rate_s2d']

# Sack rate differentials
games['sack_rate_last3_diff'] = games['home_sack_rate_last3'] - games['away_sack_rate_last3']
games['sack_rate_last5_diff'] = games['home_sack_rate_last5'] - games['away_sack_rate_last5']
games['sack_rate_s2d_diff'] = games['home_sack_rate_s2d'] - games['away_sack_rate_s2d']

# Explosive play rate differentials
games['explosive_rate_last3_diff'] = games['home_explosive_rate_last3'] - games['away_explosive_rate_last3']
games['explosive_rate_last5_diff'] = games['home_explosive_rate_last5'] - games['away_explosive_rate_last5']
games['explosive_rate_s2d_diff'] = games['home_explosive_rate_s2d'] - games['away_explosive_rate_s2d']

# Red zone touchdown rate differentials
games['red_zone_td_rate_last3_diff'] = games['home_red_zone_td_rate_last3'] - games['away_red_zone_td_rate_last3']
games['red_zone_td_rate_last5_diff'] = games['home_red_zone_td_rate_last5'] - games['away_red_zone_td_rate_last5']
games['red_zone_td_rate_s2d_diff'] = games['home_red_zone_td_rate_s2d'] - games['away_red_zone_td_rate_s2d']

# Pace differentials
games['plays_per_game_last3_diff'] = games['home_plays_per_game_last3'] - games['away_plays_per_game_last3']
games['plays_per_game_last5_diff'] = games['home_plays_per_game_last5'] - games['away_plays_per_game_last5']
games['plays_per_game_s2d_diff'] = games['home_plays_per_game_s2d'] - games['away_plays_per_game_s2d']

# Rest differential
games['rest_diff'] = games['home_rest'] - games['away_rest']

# Drop tied games (rare, ~0.4% of data) since binary win/loss target can't represent a tie
games = games[games['home_points_for'] != games['away_points_for']]
games['home_team_win'] = (games['home_points_for'] > games['away_points_for']).astype(int)

diff_cols = [col for col in games.columns if col.endswith('_diff')]

# Fill remaining differential NaNs (Week 1 of each season, or 2016's first
# few weeks) with 0. Treated as "no signal yet," not a real team-quality
# assumption.
games[diff_cols] = games[diff_cols].fillna(0)

# Final feature set for Phase 1 logistic regression: rolling home-minus-away
# differentials + context flags + target. Raw (non-rolling) differentials
# are deliberately excluded here to prevent leakage (see note above).
modeling_cols = [
    'home_season', 'home_week', 'home_team', 'away_team',
    'div_game',
    'home_short_week', 'away_short_week', 'home_bye', 'away_bye',
    'off_epa_last3_diff', 'off_epa_last5_diff', 'off_epa_s2d_diff',
    'def_epa_last3_diff', 'def_epa_last5_diff', 'def_epa_s2d_diff',
    'off_success_last3_diff', 'off_success_last5_diff', 'off_success_s2d_diff',
    'def_success_last3_diff', 'def_success_last5_diff', 'def_success_s2d_diff',
    'point_diff_last3_diff', 'point_diff_last5_diff', 'point_diff_s2d_diff',
    'turnover_margin_last3_diff', 'turnover_margin_last5_diff', 'turnover_margin_s2d_diff',
    'explosive_rate_last3_diff', 'explosive_rate_last5_diff', 'explosive_rate_s2d_diff',
    'third_down_rate_last3_diff', 'third_down_rate_last5_diff', 'third_down_rate_s2d_diff',
    'sack_rate_last3_diff', 'sack_rate_last5_diff', 'sack_rate_s2d_diff',
    'red_zone_td_rate_last3_diff', 'red_zone_td_rate_last5_diff', 'red_zone_td_rate_s2d_diff',
    'plays_per_game_last3_diff', 'plays_per_game_last5_diff', 'plays_per_game_s2d_diff',
    'rest_diff',
    'home_team_win'
]

modeling_table = games[modeling_cols].copy()
modeling_table = modeling_table.rename(columns={'home_season': 'season', 'home_week': 'week'})
modeling_table.to_csv("data/processed/modeling_table.csv", index=False)
print(f"Saved modeling_table.csv ({len(modeling_table)} games, {len(modeling_table.columns)} columns)")
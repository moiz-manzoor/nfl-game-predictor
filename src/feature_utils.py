"""
feature_utils.py

Shared helper functions for building rolling, leakage-safe team-week
features. Each build_* function takes a raw aggregated stat (e.g. total
turnovers per team-week) and returns it with a rate/margin calculated plus
last3/last5/season-to-date rolling versions via apply_rolling_windows.

These helpers are used by both features.py (the training pipeline, which
processes the full historical dataset) and predict_week.py (the prediction
pipeline, which recomputes the same features on-demand for a target week).
Keeping the logic here in one place ensures both pipelines build features
identically.
"""

import pandas as pd


def apply_rolling_windows(df, stat_col, team_col='team', season_col='season'):
    """
    Apply leakage-safe rolling windows (last3, last5, season-to-date) to a stat column.

    Each row's rolling value only includes strictly prior games via shift(1)
    before the rolling/expanding window, so a team's Week N features never
    include Week N's own game.
    """
    df = df.sort_values([team_col, season_col, 'week'])
    df[f'{stat_col}_last3'] = df.groupby(team_col)[stat_col].transform(lambda x: x.shift(1).rolling(3).mean())
    df[f'{stat_col}_last5'] = df.groupby(team_col)[stat_col].transform(lambda x: x.shift(1).rolling(5).mean())
    df[f'{stat_col}_s2d'] = df.groupby([team_col, season_col])[stat_col].transform(lambda x: x.shift(1).expanding().mean())
    return df


def build_turnover_margin(turnovers_committed, turnovers_forced):
    """Combine turnovers committed/forced into a per-team-week margin, with rolling windows."""
    turnovers = turnovers_committed.merge(turnovers_forced, on=['team', 'season', 'week'])
    turnovers['turnover_margin'] = turnovers['turnovers_forced'] - turnovers['turnovers_committed']
    return apply_rolling_windows(turnovers, 'turnover_margin')


def build_third_down_rate(third_down):
    """Calculate third-down conversion rate per team-week, with rolling windows."""
    third_down['third_down_rate'] = third_down['conversions'] / third_down['attempts']
    third_down = third_down.rename(columns={'posteam': 'team'})
    return apply_rolling_windows(third_down, 'third_down_rate')


def build_explosive_rate(explosive_plays):
    """Calculate explosive play rate (10+ yd runs, 15+ yd passes) per team-week, with rolling windows."""
    explosive_plays['explosive_rate'] = explosive_plays['explosive_plays'] / explosive_plays['total_plays']
    explosive_plays = explosive_plays.rename(columns={'posteam': 'team'})
    return apply_rolling_windows(explosive_plays, 'explosive_rate')


def build_red_zone_rate(red_zone):
    """Calculate red zone touchdown rate per team-week, with rolling windows."""
    red_zone['red_zone_td_rate'] = red_zone['red_zone_tds'] / red_zone['red_zone_trips']
    red_zone = red_zone.rename(columns={'posteam': 'team'})
    return apply_rolling_windows(red_zone, 'red_zone_td_rate')


def build_pace(pbp_filtered):
    """Calculate plays per game (pace) per team-week, with rolling windows."""
    pace_stats = pbp_filtered.groupby(['posteam', 'week', 'season'])['play_id'].count().reset_index()
    pace_stats = pace_stats.rename(columns={'play_id': 'plays_per_game', 'posteam': 'team'})
    return apply_rolling_windows(pace_stats, 'plays_per_game')


def build_sack_rate(pbp_filtered):
    """Calculate sack rate per team-week, with rolling windows."""
    sack_stats = pbp_filtered.groupby(['posteam', 'week', 'season'])['sack'].mean().reset_index()
    sack_stats = sack_stats.rename(columns={'sack': 'sack_rate', 'posteam': 'team'})
    return apply_rolling_windows(sack_stats, 'sack_rate')
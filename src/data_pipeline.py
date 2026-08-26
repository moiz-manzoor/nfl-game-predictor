"""
data_pipeline.py

Pulls raw NFL data from nflreadpy and saves it to data/raw/ as CSVs:
- schedules.csv   : game-level schedule info, scores, rest days, moneylines
- team_stats.csv  : official team-level stats (kept for reference; not the
                    primary source for EPA, see note below)
- pbp.csv         : play-by-play data, the main source for engineered features

Two separate season ranges are used:
- SCHEDULE_SEASONS includes the upcoming season, since the NFL schedule
  (matchups, dates) is published months before games are played.
- STATS_SEASONS stops at the last fully completed season, since team stats
  and play-by-play data only exist for games that have actually been played.
  Requesting a not-yet-played season here will 404.
"""

import nflreadpy as nfl

SCHEDULE_SEASONS = list(range(2016, 2027))  # includes 2026 for upcoming schedule
STATS_SEASONS = list(range(2016, 2026))     # stops at 2025, last season with played games


def pull_schedules(seasons):
    """Load game-level schedule data (matchups, scores, rest days, moneylines)."""
    schedules = nfl.load_schedules(seasons=seasons)
    return schedules.to_pandas()


def pull_team_stats(seasons):
    """Load official team-level stats. Not used for EPA (see pull_pbp)."""
    team_stats = nfl.load_team_stats(seasons=seasons)
    return team_stats.to_pandas()


def pull_pbp(seasons):
    """
    Load play-by-play data.

    team_stats lacks defensive EPA and a unified offensive EPA (it only
    splits passing/rushing/receiving separately). Play-by-play lets us
    calculate both offensive and defensive EPA/play ourselves using the
    posteam/defteam columns.
    """
    pbp = nfl.load_pbp(seasons=seasons)
    return pbp.to_pandas()


if __name__ == "__main__":
    print("Script started")
    schedules_df = pull_schedules(SCHEDULE_SEASONS)
    print("Schedules pulled")
    team_stats_df = pull_team_stats(STATS_SEASONS)
    print("Team stats pulled")
    pbp_df = pull_pbp(STATS_SEASONS)
    print("Play-by-play data pulled")

    schedules_df.to_csv("data/raw/schedules.csv", index=False)
    team_stats_df.to_csv("data/raw/team_stats.csv", index=False)
    pbp_df.to_csv("data/raw/pbp.csv", index=False)
    print(f"Data saved to CSV files ({schedules_df.shape[0]} schedule rows, "
          f"{team_stats_df.shape[0]} team-stat rows, {pbp_df.shape[0]} pbp rows)")
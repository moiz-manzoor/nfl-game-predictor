"""
  Automated leakage tests for the feature pipeline (src/features.py).
  Independently recalculates rolling/expanding EPA values from raw
  play-by-play data and asserts they match the saved feature table —
  confirming no future-game data leaks into any team-week's features.
"""
import pandas as pd
import numpy as np

features = pd.read_csv("data/processed/team_week_features.csv")
#print(features.shape)
#print(features.columns.tolist())

# Load raw pbp data 
pbp = pd.read_csv("data/raw/pbp.csv", low_memory=False)
pbp_filtered = pbp[pbp['play_type'].isin(['pass', 'run'])]

# Get the value we are testing
test_row = features[(features['team'] == 'ARI') & (features['season'] == 2016) & (features['week'] == 6)]
stored_value = test_row['off_epa_last5'].values[0]
print("stored value:", stored_value)

# Recalculate value from raw pbp data independently from features.py
arizona_weeks_1to5 = pbp_filtered[
    (pbp_filtered['posteam'] == 'ARI') &
    (pbp_filtered['season'] == 2016) &
    (pbp_filtered['week'] >= 1) &
    (pbp_filtered['week'] <= 5)
]

recalculated_value = arizona_weeks_1to5.groupby('week')['epa'].mean().mean()
print("recalculated value:", recalculated_value)

assert np.isclose(stored_value, recalculated_value), "off_epa_last5 does not match the independent recalculation!"

arizona_def_weeks_1to5 = pbp_filtered[
    (pbp_filtered['defteam'] == 'ARI') &
    (pbp_filtered['season'] == 2016) &
    (pbp_filtered['week'] >= 1) &
    (pbp_filtered['week'] <= 5)
]

stored_def_epa_last5 = test_row['def_epa_last5'].values[0]
recalculated_def_epa_last5 = arizona_def_weeks_1to5.groupby('week')['epa'].mean().mean()

print("stored def_epa_last5:", stored_def_epa_last5)
print("recalculated def_epa_last5:", recalculated_def_epa_last5)

assert np.isclose(stored_def_epa_last5, recalculated_def_epa_last5), "def_epa_last5 does not match independent recalculation!"


buf_test_row = features[(features['team'] == 'BUF') & (features['season'] == 2016) & (features['week'] == 4)]
stored_off_epa_s2d = buf_test_row['off_epa_s2d'].values[0]

buf_weeks_1_to_3 = pbp_filtered[
    (pbp_filtered['posteam'] == 'BUF') &
    (pbp_filtered['season'] == 2016) &
    (pbp_filtered['week'] >= 1) &
    (pbp_filtered['week'] <= 3)
]

recalculated_off_epa_s2d = buf_weeks_1_to_3.groupby('week')['epa'].mean().mean()

print("stored off_epa_s2d:", stored_off_epa_s2d)
print("recalculated off_epa_s2d:", recalculated_off_epa_s2d)

assert np.isclose(stored_off_epa_s2d, recalculated_off_epa_s2d), "off_epa_s2d does not match independent recalculation!"

print("\nAll leakage tests passed.")
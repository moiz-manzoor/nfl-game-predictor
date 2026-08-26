"""
combine_predictions.py

Merges Model A and Model B's saved predictions for the same week into a
single side-by-side comparison CSV, including Vegas implied probability
and a flag for whether the two models agree on the winner.

Assumes both models' predictions for the target week have already been
generated via predict_week.py (--model a and --model b).
"""

import pandas as pd

model_a = pd.read_csv("outputs/predictions/week1_2026_modela_predictions.csv")
model_b = pd.read_csv("outputs/predictions/week1_2026_modelb_predictions.csv")

comparison = model_a[['home_team', 'away_team', 'home_win_prob', 'predicted_winner', 'home_implied_prob']].merge(
    model_b[['home_team', 'away_team', 'home_win_prob', 'predicted_winner']],
    on=['home_team', 'away_team'],
    suffixes=('_model_a', '_model_b')
)

comparison = comparison.rename(columns={
    'home_win_prob_model_a': 'model_a_home_win_prob',
    'predicted_winner_model_a': 'model_a_pick',
    'home_win_prob_model_b': 'model_b_home_win_prob',
    'predicted_winner_model_b': 'model_b_pick',
    'home_implied_prob': 'vegas_home_implied_prob'
})

comparison['models_agree'] = comparison['model_a_pick'] == comparison['model_b_pick']

comparison = comparison[['home_team', 'away_team',
                          'model_a_home_win_prob', 'model_a_pick',
                          'model_b_home_win_prob', 'model_b_pick',
                          'vegas_home_implied_prob', 'models_agree']]

comparison.to_csv("outputs/predictions/week1_2026_model_comparison.csv", index=False)
print(comparison)
print(f"\nSaved outputs/predictions/week1_2026_model_comparison.csv ({len(comparison)} games)")
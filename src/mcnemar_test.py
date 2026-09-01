"""
mcnemar_test.py

Runs McNemar's test comparing Model A against Vegas on the same 2025
holdout games, to put a real statistical significance number behind the
~2-point accuracy gap flagged in the README's Limitations section.

Why McNemar's test specifically: Model A and Vegas are evaluated on the
exact same 270 games, which makes this a paired comparison, not two
independent samples. McNemar's test is built for exactly this case, two
classifiers, same test set, binary correct/incorrect outcome. It only looks
at the games where the model and Vegas disagreed (one right, one wrong) and
tests whether that disagreement pattern is asymmetric enough to be
statistically significant, rather than comparing raw accuracy numbers
directly.

Requires: pip install statsmodels

Outputs: prints the 2x2 contingency table, the McNemar statistic, and the
p-value, with a plain-language interpretation.
"""

import joblib
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def moneyline_to_prob(moneyline):
    """Convert an American moneyline to a raw (un-de-vigged) implied probability."""
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)


# --- Load Model A and the same 2025 test set used in train_classifier.py ---
model_a = joblib.load('models/model_a.pkl')
scaler_a = joblib.load('models/scaler_a.pkl')

modeling_table = pd.read_csv("data/processed/modeling_table.csv")
test_a = modeling_table[modeling_table['season'] == 2025]

feature_cols = [col for col in modeling_table.columns if col.endswith('_diff')]
x_test_a = test_a[feature_cols]
y_test_a = test_a['home_team_win']

x_test_a_scaled = scaler_a.transform(x_test_a)
y_pred_a = model_a.predict(x_test_a_scaled)

# Convert to numpy arrays immediately so later comparisons align by position,
# not by pandas index label (test_a keeps modeling_table's original row
# numbers, which don't match up with the merge result's fresh index below).
y_pred_a = y_pred_a.reshape(-1)
y_test_a_arr = y_test_a.to_numpy()
model_correct = (y_pred_a == y_test_a_arr).astype(int)

# --- Rebuild Vegas predictions the same way as train_classifier.py ---
schedules = pd.read_csv("data/raw/schedules.csv")

schedules['home_implied_prob_raw'] = schedules['home_moneyline'].apply(moneyline_to_prob)
schedules['away_implied_prob_raw'] = schedules['away_moneyline'].apply(moneyline_to_prob)
schedules['vig_total'] = schedules['home_implied_prob_raw'] + schedules['away_implied_prob_raw']
schedules['home_implied_prob'] = schedules['home_implied_prob_raw'] / schedules['vig_total']

vegas_probs = schedules[['season', 'week', 'home_team', 'away_team', 'home_implied_prob']].dropna(subset=['home_implied_prob'])

test_a_vegas = test_a.merge(
    vegas_probs,
    on=['season', 'week', 'home_team', 'away_team'],
    how='left'
)

# Left merge preserves row order, but only if no key produced duplicate
# matches. Confirm row count didn't change before trusting positional
# alignment with model_correct.
assert len(test_a_vegas) == len(test_a), (
    f"Merge changed row count ({len(test_a)} -> {len(test_a_vegas)}), "
    f"check for duplicate keys in vegas_probs before trusting positional alignment"
)

vegas_pred = (test_a_vegas['home_implied_prob'] > 0.5).astype(int).to_numpy()
vegas_correct = (vegas_pred == test_a_vegas['home_team_win'].to_numpy()).astype(int)

# --- Sanity check: make sure both prediction arrays line up game-for-game ---
assert len(model_correct) == len(vegas_correct), "Model A and Vegas prediction counts don't match, check merges"

# --- Build the 2x2 contingency table for McNemar's test ---
both_correct = ((model_correct == 1) & (vegas_correct == 1)).sum()
model_only = ((model_correct == 1) & (vegas_correct == 0)).sum()
vegas_only = ((model_correct == 0) & (vegas_correct == 1)).sum()
both_wrong = ((model_correct == 0) & (vegas_correct == 0)).sum()

table = [[both_correct, model_only],
         [vegas_only, both_wrong]]

print("2025 Model A vs Vegas — paired outcomes:")
print(f"  Both correct:        {both_correct}")
print(f"  Model right, Vegas wrong: {model_only}")
print(f"  Vegas right, Model wrong: {vegas_only}")
print(f"  Both wrong:           {both_wrong}")
print(f"  Total games:          {len(model_correct)}")

# --- Run McNemar's test on the discordant pairs (model_only vs vegas_only) ---
discordant_total = model_only + vegas_only
use_exact = discordant_total < 25  # small-sample rule of thumb

result = mcnemar(table, exact=use_exact, correction=not use_exact)

print(f"\nMcNemar's test ({'exact binomial' if use_exact else 'chi-square with continuity correction'}):")
print(f"  Statistic: {result.statistic:.4f}")
print(f"  p-value:   {result.pvalue:.4f}")

alpha = 0.05
if result.pvalue < alpha:
    print(f"\nResult: statistically significant at alpha={alpha}. "
          f"The gap between Model A and Vegas is unlikely to be due to chance alone.")
else:
    print(f"\nResult: not statistically significant at alpha={alpha}. "
          f"The ~2-point gap could plausibly be due to chance given this sample size, "
          f"consistent with the 'directional, not confirmed' framing in the README.")
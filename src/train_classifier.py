"""
train_classifier.py

Trains two logistic regression models on the modeling table produced by
features.py, using two different time-based train/test splits:

- Model A: trained on seasons <=2024, tested on 2025 only
- Model B: trained on seasons <=2023, tested on 2024-2025

Both models use the same feature set (all home-minus-away rolling
differentials) and the same regularization strength (C=0.01, selected via
a C-value sweep that showed stronger regularization outperforming the
default C=1.0 on both splits).

Outputs:
- reports/feature_importance_a.png, feature_importance_b.png
- reports/calibration_plot_a.png, calibration_plot_b.png
- reports/calibration_comparison.png (Model A vs Vegas)
- models/model_a.pkl, scaler_a.pkl, model_b.pkl, scaler_b.pkl
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_score, recall_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler


def moneyline_to_prob(moneyline):
    """Convert an American moneyline to a raw (un-de-vigged) implied probability."""
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)


# --- Load modeling table ---
modeling_table = pd.read_csv("data/processed/modeling_table.csv")

# --- Time-based splits: Split A (test=2025), Split B (test=2024-2025) ---
train_a = modeling_table[modeling_table['season'] <= 2024]
test_a = modeling_table[modeling_table['season'] == 2025]

train_b = modeling_table[modeling_table['season'] <= 2023]
test_b = modeling_table[modeling_table['season'] >= 2024]

feature_cols = [col for col in modeling_table.columns if col.endswith('_diff')]

# --- Split A: features/target, scaled ---
x_train_a = train_a[feature_cols]
y_train_a = train_a['home_team_win']
x_test_a = test_a[feature_cols]
y_test_a = test_a['home_team_win']

scaler_a = StandardScaler()
x_train_a_scaled = scaler_a.fit_transform(x_train_a)
x_test_a_scaled = scaler_a.transform(x_test_a)

# --- Split B: features/target, scaled ---
x_train_b = train_b[feature_cols]
y_train_b = train_b['home_team_win']
x_test_b = test_b[feature_cols]
y_test_b = test_b['home_team_win']

scaler_b = StandardScaler()
x_train_b_scaled = scaler_b.fit_transform(x_train_b)
x_test_b_scaled = scaler_b.transform(x_test_b)

# --- Train baseline logistic regression on both splits ---
model_a = LogisticRegression(max_iter=1000, C=0.01)
model_a.fit(x_train_a_scaled, y_train_a)
model_b = LogisticRegression(max_iter=1000, C=0.01)
model_b.fit(x_train_b_scaled, y_train_b)

# --- Feature importance: Model A coefficients ranked by magnitude ---
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model_a.coef_[0]
})
print("Model A feature coefficients (sorted by magnitude):")
print(coef_df.sort_values('coefficient', key=abs, ascending=False))

top_features = coef_df.reindex(coef_df['coefficient'].abs().sort_values(ascending=False).index).head(20)

plt.figure(figsize=(10, 8))
colors = ['#d62728' if c < 0 else '#1f77b4' for c in top_features['coefficient']]
plt.barh(top_features['feature'], top_features['coefficient'], color=colors)
plt.xlabel('Coefficient (standardized)')
plt.title('Top 20 Feature Coefficients - Model A')
plt.gca().invert_yaxis()
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('reports/feature_importance_a.png')
plt.show()

# --- Feature importance: Model B coefficients ranked by magnitude ---
coef_df_b = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model_b.coef_[0]
})
print("Model B feature coefficients (sorted by magnitude):")
print(coef_df_b.sort_values('coefficient', key=abs, ascending=False))

top_features_b = coef_df_b.reindex(coef_df_b['coefficient'].abs().sort_values(ascending=False).index).head(20)

plt.figure(figsize=(10, 8))
colors_b = ['#d62728' if c < 0 else '#1f77b4' for c in top_features_b['coefficient']]
plt.barh(top_features_b['feature'], top_features_b['coefficient'], color=colors_b)
plt.xlabel('Coefficient (standardized)')
plt.title('Top 20 Feature Coefficients - Model B')
plt.gca().invert_yaxis()
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('reports/feature_importance_b.png')
plt.show()

# --- Predictions on held-out test sets ---
y_pred_a = model_a.predict(x_test_a_scaled)
y_pred_proba_a = model_a.predict_proba(x_test_a_scaled)[:, 1]

y_pred_b = model_b.predict(x_test_b_scaled)
y_pred_proba_b = model_b.predict_proba(x_test_b_scaled)[:, 1]

# --- Evaluate: accuracy, log loss, confusion matrix, precision/recall ---
accuracy_a = accuracy_score(y_test_a, y_pred_a)
accuracy_b = accuracy_score(y_test_b, y_pred_b)
log_loss_a = log_loss(y_test_a, y_pred_proba_a)
log_loss_b = log_loss(y_test_b, y_pred_proba_b)
cm_a = confusion_matrix(y_test_a, y_pred_a)
cm_b = confusion_matrix(y_test_b, y_pred_b)
precision_a = precision_score(y_test_a, y_pred_a)
recall_a = recall_score(y_test_a, y_pred_a)
precision_b = precision_score(y_test_b, y_pred_b)
recall_b = recall_score(y_test_b, y_pred_b)

print(f"Model A — accuracy: {accuracy_a:.4f}, log loss: {log_loss_a:.4f}, precision: {precision_a:.4f}, recall: {recall_a:.4f}")
print("Model A confusion matrix:")
print(cm_a)
print(f"Model B — accuracy: {accuracy_b:.4f}, log loss: {log_loss_b:.4f}, precision: {precision_b:.4f}, recall: {recall_b:.4f}")
print("Model B confusion matrix:")
print(cm_b)

# --- Calibration plots: verify predicted probabilities match actual outcomes ---
prob_true_a, prob_pred_a = calibration_curve(y_test_a, y_pred_proba_a, n_bins=10)
plt.figure()
plt.plot(prob_pred_a, prob_true_a, marker='o', label='Model A')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Predicted probability of home win')
plt.ylabel('Actual fraction of home wins')
plt.title('Calibration Plot — Model A')
plt.legend()
plt.savefig('reports/calibration_plot_a.png')
plt.show()

prob_true_b, prob_pred_b = calibration_curve(y_test_b, y_pred_proba_b, n_bins=10)
plt.figure()
plt.plot(prob_pred_b, prob_true_b, marker='o', label='Model B', color='orange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Predicted probability of home win')
plt.ylabel('Actual fraction of home wins')
plt.title('Calibration Plot — Model B')
plt.legend()
plt.savefig('reports/calibration_plot_b.png')
plt.show()

# --- Compare Model A predictions to Vegas implied probabilities ---
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

vegas_pred = (test_a_vegas['home_implied_prob'] > 0.5).astype(int)
vegas_accuracy = accuracy_score(test_a_vegas['home_team_win'], vegas_pred)
vegas_log_loss = log_loss(test_a_vegas['home_team_win'], test_a_vegas['home_implied_prob'])

print(f"Vegas accuracy: {vegas_accuracy:.4f}, log loss: {vegas_log_loss:.4f}")
print(f"Model A accuracy: {accuracy_a:.4f}, log loss: {log_loss_a:.4f}")

vegas_prob_true, vegas_prob_pred = calibration_curve(
    test_a_vegas['home_team_win'], test_a_vegas['home_implied_prob'], n_bins=10
)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred_a, prob_true_a, marker='o', label='Model A', color='#1f77b4')
plt.plot(vegas_prob_pred, vegas_prob_true, marker='s', label='Vegas', color='#2ca02c')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Predicted probability of home win')
plt.ylabel('Actual fraction of home wins')
plt.title('Calibration Comparison — Model A vs Vegas')
plt.legend()
plt.tight_layout()
plt.savefig('reports/calibration_comparison.png')
plt.show()

# --- Save trained models and scalers for use in predict_week.py ---
joblib.dump(model_a, 'models/model_a.pkl')
joblib.dump(scaler_a, 'models/scaler_a.pkl')
print("Saved model_a and scaler_a")

joblib.dump(model_b, 'models/model_b.pkl')
joblib.dump(scaler_b, 'models/scaler_b.pkl')
print("Saved model_b and scaler_b")
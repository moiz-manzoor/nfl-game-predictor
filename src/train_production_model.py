"""
train_production_model.py

Trains a production logistic regression model on ALL available seasons
(2016-2025), with no holdout, for the purpose of predicting live 2026 games.

This is intentionally separate from train_classifier.py. Model A and Model B
in train_classifier.py are evaluation models: their holdout seasons (2025,
and 2024-2025 respectively) exist specifically to produce trustworthy,
reported accuracy numbers (64.4% / 64.7%, documented in the README and
resume). Retraining on all data through 2025 means recent games actually
inform the coefficients used for real 2026 predictions, but it also means
this model has no fresh, comparable holdout accuracy number, since none of
its training data was withheld. That's an expected tradeoff for a
deployment model, not a flaw.

Same feature set and regularization strength (C=0.01) as Model A/B, so the
only thing that changes is the training window.

Outputs:
- models/model_a_2026.pkl, models/scaler_a_2026.pkl
- reports/feature_importance_a_2026.png

Does NOT touch or overwrite model_a.pkl, model_b.pkl, or their existing
scalers/reports.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load modeling table 
modeling_table = pd.read_csv("data/processed/modeling_table.csv")

# No holdout: train on every available season (2016-2025) 
train_production = modeling_table[modeling_table['season'] <= 2025]

feature_cols = [col for col in modeling_table.columns if col.endswith('_diff')]

x_train_production = train_production[feature_cols]
y_train_production = train_production['home_team_win']

scaler_production = StandardScaler()
x_train_production_scaled = scaler_production.fit_transform(x_train_production)

# Train logistic regression, same C=0.01 as Model A/B
model_production = LogisticRegression(max_iter=1000, C=0.01)
model_production.fit(x_train_production_scaled, y_train_production)

print(f"Production model trained on {len(train_production)} games (seasons 2016-2025, no holdout)")

# Feature importance, kept for reference. No test-set accuracy exists to pair with it.
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model_production.coef_[0]
})
print("Production model feature coefficients (sorted by magnitude):")
print(coef_df.sort_values('coefficient', key=abs, ascending=False))

top_features = coef_df.reindex(coef_df['coefficient'].abs().sort_values(ascending=False).index).head(20)

plt.figure(figsize=(10, 8))
colors = ['#d62728' if c < 0 else '#1f77b4' for c in top_features['coefficient']]
plt.barh(top_features['feature'], top_features['coefficient'], color=colors)
plt.xlabel('Coefficient (standardized)')
plt.title('Top 20 Feature Coefficients - Production Model (2016-2025, no holdout)')
plt.gca().invert_yaxis()
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('reports/feature_importance_a_2026.png')
plt.show()

# Save trained model and scaler for use in predict_week.py
joblib.dump(model_production, 'models/model_a_2026.pkl')
joblib.dump(scaler_production, 'models/scaler_a_2026.pkl')
print("Saved model_a_2026 and scaler_a_2026")
print("Note: this model has no holdout accuracy number by design, all available")
print("data (2016-2025) went into training it for live 2026 deployment.")
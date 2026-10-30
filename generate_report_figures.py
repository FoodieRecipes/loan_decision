import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Set aesthetic publication styling
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

base_dir = r"c:\Users\USER\Desktop\Goody_Project"
fig_dir = os.path.join(base_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# Load metrics
with open(os.path.join(base_dir, "models", "metrics.json"), "r") as f:
    metrics = json.load(f)

# Load dataset and preprocessor
df = pd.read_csv(os.path.join(base_dir, "nigeria_credit_risk_dataset_2500.csv"))
preprocessor = joblib.load(os.path.join(base_dir, "models", "preprocessor.joblib"))
rf_model = joblib.load(os.path.join(base_dir, "models", "rf_risk_model.joblib"))
xgb_model = joblib.load(os.path.join(base_dir, "models", "xgb_risk_model.joblib"))
rf_bin = joblib.load(os.path.join(base_dir, "models", "rf_binary_model.joblib"))
xgb_bin = joblib.load(os.path.join(base_dir, "models", "xgb_binary_model.joblib"))

numeric_features = metrics['features']['numeric']
categorical_features = metrics['features']['categorical']

risk_map = {'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}
y_risk = df['risk_level'].map(risk_map).values
y_binary = df['is_default'].values

X_processed = preprocessor.transform(df[numeric_features + categorical_features])

# Recreate the exact 80/20 split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train_risk, y_test_risk, y_train_bin, y_test_bin = train_test_split(
    X_processed, y_risk, y_binary, test_size=0.20, random_state=42, stratify=y_risk
)

# Predictions
rf_proba = rf_model.predict_proba(X_test)
xgb_proba = xgb_model.predict_proba(X_test)
rf_bin_proba = rf_bin.predict_proba(X_test)[:, 1]
xgb_bin_proba = xgb_bin.predict_proba(X_test)[:, 1]

print("Generating figures...")

# -------------------------------------------------------------
# FIG 4.3: METRICS BENCHMARK
# -------------------------------------------------------------
plt.figure(figsize=(10, 5.5), dpi=300)
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Multi-Class AUC', 'Binary Default AUC']
rf_vals = [94.80, 95.01, 94.80, 94.86, 99.59, 99.81]
xgb_vals = [95.20, 95.12, 95.20, 95.12, 99.57, 99.80]

x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, rf_vals, width, label='Tuned Random Forest', color='#6366f1', edgecolor='black', alpha=0.9)
plt.bar(x + width/2, xgb_vals, width, label='Tuned XGBoost (Selected Best)', color='#0d9488', edgecolor='black', alpha=0.9)

for i in range(len(labels)):
    plt.text(x[i] - width/2, rf_vals[i] + 0.4, f"{rf_vals[i]:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#312e81')
    plt.text(x[i] + width/2, xgb_vals[i] + 0.4, f"{xgb_vals[i]:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0f766e')

plt.title("Comparative Performance Metrics Benchmark: Tuned Random Forest vs. Tuned XGBoost", fontweight='bold', pad=14)
plt.ylabel("Score (%)")
plt.xticks(x, labels, fontweight='bold')
plt.ylim(88, 102)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig4_3_metrics_comparison.png"))
plt.close()

# -------------------------------------------------------------
# FIG 4.4: CV VS TEST GENERALIZATION
# -------------------------------------------------------------
plt.figure(figsize=(9, 5.2), dpi=300)
models = ['Random Forest', 'XGBoost']
cv_scores = [95.75, 95.55]
test_scores = [94.80, 95.20]

x = np.arange(len(models))
width = 0.30

plt.bar(x - width/2, cv_scores, width, label='3-Fold Stratified Cross-Validation', color='#64748b', edgecolor='black')
plt.bar(x + width/2, test_scores, width, label='Unseen Holdout Test Set (500 Loans)', color='#0d9488', edgecolor='black')

for i in range(len(models)):
    plt.text(x[i] - width/2, cv_scores[i] + 0.2, f"{cv_scores[i]:.2f}%", ha='center', fontweight='bold', fontsize=10)
    plt.text(x[i] + width/2, test_scores[i] + 0.2, f"{test_scores[i]:.2f}%", ha='center', fontweight='bold', fontsize=10)

plt.title("Cross-Validation vs. Holdout Test Generalization (Overfitting Audit)", fontweight='bold', pad=14)
plt.ylabel("Classification Accuracy (%)")
plt.xticks(x, models, fontweight='bold', fontsize=11)
plt.ylim(90, 98)
plt.legend(loc='lower left', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig4_4_cv_vs_test.png"))
plt.close()

# -------------------------------------------------------------
# FIG 4.5: SIDE-BY-SIDE CONFUSION MATRICES
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
classes = ['Low Risk', 'Medium Risk', 'High Risk']

cm_rf = np.array(metrics['random_forest']['confusion_matrix'])
cm_xgb = np.array(metrics['xgboost']['confusion_matrix'])

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=classes, yticklabels=classes, annot_kws={"size": 13, "weight": "bold"})
axes[0].set_title("Tuned Random Forest Confusion Matrix\n(Accuracy: 94.80% | F1: 94.86%)", fontweight='bold')
axes[0].set_xlabel("Predicted Risk Tier")
axes[0].set_ylabel("Actual Risk Tier")

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
            xticklabels=classes, yticklabels=classes, annot_kws={"size": 13, "weight": "bold"})
axes[1].set_title("Tuned XGBoost Confusion Matrix\n(Accuracy: 95.20% | F1: 95.12%)", fontweight='bold')
axes[1].set_xlabel("Predicted Risk Tier")
axes[1].set_ylabel("Actual Risk Tier")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig4_5_confusion_matrices.png"))
plt.close()

# -------------------------------------------------------------
# FIG 4.6: PER-CLASS F1-SCORE AND RECALL
# -------------------------------------------------------------
plt.figure(figsize=(10, 5.2), dpi=300)
risk_tiers = ['Low Risk', 'Medium Risk', 'High Risk']
rf_f1_cls = [96.89, 86.32, 96.62]
xgb_f1_cls = [97.34, 86.36, 96.64]

x = np.arange(len(risk_tiers))
width = 0.32

plt.bar(x - width/2, rf_f1_cls, width, label='Random Forest F1-Score', color='#6366f1', edgecolor='black')
plt.bar(x + width/2, xgb_f1_cls, width, label='XGBoost F1-Score', color='#0d9488', edgecolor='black')

for i in range(len(risk_tiers)):
    plt.text(x[i] - width/2, rf_f1_cls[i] + 0.4, f"{rf_f1_cls[i]:.2f}%", ha='center', fontsize=9.5, fontweight='bold')
    plt.text(x[i] + width/2, xgb_f1_cls[i] + 0.4, f"{xgb_f1_cls[i]:.2f}%", ha='center', fontsize=9.5, fontweight='bold')

plt.title("Per-Class F1-Score Across Credit Risk Tiers (Holdout Test Set)", fontweight='bold', pad=14)
plt.ylabel("F1-Score (%)")
plt.xticks(x, risk_tiers, fontweight='bold')
plt.ylim(75, 103)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig4_6_per_class_f1.png"))
plt.close()

# -------------------------------------------------------------
# FIG 4.7: BINARY DEFAULT ROC & PR CURVES
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Binary ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test_bin, rf_bin_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test_bin, xgb_bin_proba)
auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)

axes[0].plot(fpr_rf, tpr_rf, color='#6366f1', lw=2.5, label=f'Random Forest (AUC = {auc_rf:.4f})')
axes[0].plot(fpr_xgb, tpr_xgb, color='#0d9488', lw=2.5, linestyle='--', label=f'XGBoost (AUC = {auc_xgb:.4f})')
axes[0].plot([0, 1], [0, 1], color='#94a3b8', linestyle=':', label='Random Chance')
axes[0].set_title("Binary Default ROC Curves (Non-Performing Loan Discrimination)", fontweight='bold')
axes[0].set_xlabel("False Positive Rate (Type I Error)")
axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
axes[0].legend(loc='lower right')

# Precision-Recall
p_rf, r_rf, _ = precision_recall_curve(y_test_bin, rf_bin_proba)
p_xgb, r_xgb, _ = precision_recall_curve(y_test_bin, xgb_bin_proba)
ap_rf = average_precision_score(y_test_bin, rf_bin_proba)
ap_xgb = average_precision_score(y_test_bin, xgb_bin_proba)

axes[1].plot(r_rf, p_rf, color='#6366f1', lw=2.5, label=f'Random Forest (AP = {ap_rf:.4f})')
axes[1].plot(r_xgb, p_xgb, color='#0d9488', lw=2.5, linestyle='--', label=f'XGBoost (AP = {ap_xgb:.4f})')
axes[1].set_title("Precision-Recall Curves for Default Prediction", fontweight='bold')
axes[1].set_xlabel("Recall (Coverage)")
axes[1].set_ylabel("Precision (Positive Predictive Value)")
axes[1].legend(loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig4_7_roc_pr_binary.png"))
plt.close()

# -------------------------------------------------------------
# FIG 4.8: MULTICLASS OvR ROC CURVES
# -------------------------------------------------------------
plt.figure(figsize=(9, 5.5), dpi=300)
colors = ['#10b981', '#f59e0b', '#ef4444']
from sklearn.preprocessing import label_binarize
y_test_bin_multi = label_binarize(y_test_risk, classes=[0, 1, 2])

for i, (cls_name, col) in enumerate(zip(classes, colors)):
    fpr_c, tpr_c, _ = roc_curve(y_test_bin_multi[:, i], xgb_proba[:, i])
    roc_c = auc(fpr_c, tpr_c)
    plt.plot(fpr_c, tpr_c, color=col, lw=2.5, label=f'{cls_name} OvR (AUC = {roc_c:.4f})')

plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle=':', label='Chance')
plt.title("XGBoost Multi-Class One-vs-Rest (OvR) ROC Curves", fontweight='bold', pad=14)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig4_8_ovr_roc_curves.png"))
plt.close()

# -------------------------------------------------------------
# FIG 4.9: FEATURE IMPORTANCE
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

rf_imp = metrics['feature_importance']['random_forest']
xgb_imp = metrics['feature_importance']['xgboost']

# Top 10 RF
rf_top10 = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)[:10]
y_rf = [k.replace('_', ' ').title() for k, v in reversed(rf_top10)]
x_rf = [v for k, v in reversed(rf_top10)]

axes[0].barh(y_rf, x_rf, color='#6366f1', edgecolor='black')
axes[0].set_title("Random Forest Top Features (Mean Decrease in Impurity)", fontweight='bold')
axes[0].set_xlabel("Relative MDI Importance Weight")

# Top 10 XGB
xgb_top10 = sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:10]
y_xgb = [k.replace('_', ' ').title() for k, v in reversed(xgb_top10)]
x_xgb = [v for k, v in reversed(xgb_top10)]

axes[1].barh(y_xgb, x_xgb, color='#0d9488', edgecolor='black')
axes[1].set_title("XGBoost Top Features (Information Gain)", fontweight='bold')
axes[1].set_xlabel("Relative Information Gain Weight")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig4_9_feature_importance.png"))
plt.close()

# -------------------------------------------------------------
# FIG 4.10: CONCORDANCE & PROBABILITY DENSITY
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Agreement Confusion Matrix between RF and XGBoost
pred_rf = rf_model.predict(X_test)
pred_xgb = xgb_model.predict(X_test)
agree_cm = np.zeros((3, 3), dtype=int)
for r, xg in zip(pred_rf, pred_xgb):
    agree_cm[r, xg] += 1

sns.heatmap(agree_cm, annot=True, fmt='d', cmap='Purples', ax=axes[0], cbar=False,
            xticklabels=classes, yticklabels=classes, annot_kws={"size": 13, "weight": "bold"})
concordance_rate = (pred_rf == pred_xgb).mean() * 100
axes[0].set_title(f"Model Inter-Rater Agreement Matrix\n(Concordance Rate: {concordance_rate:.1f}%)", fontweight='bold')
axes[0].set_xlabel("Predicted by XGBoost")
axes[0].set_ylabel("Predicted by Random Forest")

# Probability distribution
sns.histplot(rf_bin_proba, color='#6366f1', label='Random Forest', kde=True, ax=axes[1], stat="density", bins=25, alpha=0.4)
sns.histplot(xgb_bin_proba, color='#0d9488', label='XGBoost', kde=True, ax=axes[1], stat="density", bins=25, alpha=0.4)
axes[1].set_title("Calibrated Default Probability Density Comparison", fontweight='bold')
axes[1].set_xlabel("Predicted Default Probability ($P(Default)$)")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig4_10_concordance_distribution.png"))
plt.close()

print("[SUCCESS] All 8 statistical evaluation figures created and saved in figures/ successfully!")


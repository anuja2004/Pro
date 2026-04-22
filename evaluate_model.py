"""
Federated Model Performance Evaluation
=======================================
Architecture:
  Bank 1 = IEEE Credit Card (555K rows)
  Bank 2 = BAF Bank Account  (1M rows)
  Both banks trained the SAME global model via FedAvg (no holdout split was created).

Evaluation Strategy:
  Since no holdout set was reserved before training, we use CROSS-BANK evaluation:
    - How well does the global model generalize from Bank 1's patterns to Bank 2?
    - How well does it generalize from Bank 2's patterns to Bank 1?
  This is the true test of federated generalization.
  
  We also report within-bank scores (on a 20% split) so you can compare:
    within-bank  = inflated (model saw this data)
    cross-bank   = honest   (model never saw the other bank's data)

Saves all charts to results/
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

FEATURE_NAMES = [
    'address_stability', 'amount', 'bank_tenure_months', 'credit_limit',
    'credit_risk_score', 'customer_age', 'device_distinct_emails',
    'device_fraud_count', 'device_os_encoded', 'email_is_free',
    'emp_CA', 'emp_CB', 'emp_CC', 'emp_CD', 'emp_CE', 'emp_CF',
    'has_other_cards', 'house_BA', 'house_BB', 'house_BC', 'house_BD',
    'house_BE', 'house_BF', 'is_foreign_request', 'keep_alive',
    'location_velocity', 'month', 'name_email_similarity', 'phone_valid',
    'session_length', 'source_internet', 'time_since_request_days',
    'velocity_24h', 'velocity_4w', 'velocity_6h'
]

STYLE = {
    'bg':       '#0f0c29',
    'panel':    '#1a1640',
    'accent':   '#667eea',
    'fraud':    '#ff4444',
    'legit':    '#00c851',
    'flag':     '#ffaa00',
    'text':     '#ffffff',
    'subtext':  '#aaaaaa',
}

def apply_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  STYLE['bg'],
        'axes.facecolor':    STYLE['panel'],
        'axes.edgecolor':    STYLE['accent'],
        'axes.labelcolor':   STYLE['text'],
        'xtick.color':       STYLE['subtext'],
        'ytick.color':       STYLE['subtext'],
        'text.color':        STYLE['text'],
        'grid.color':        '#2a2660',
        'grid.linestyle':    '--',
        'grid.alpha':        0.5,
        'font.family':       'DejaVu Sans',
        'font.size':         11,
    })

# ── Load model ──────────────────────────────────────────────────────────────
print("\n🤖 Loading federated model...")
model = tf.keras.models.load_model('models/saved_models/federated_model_final.h5')
print(f"   Input shape: {model.input_shape}  |  Features: {model.input_shape[1]}")

# ── Load datasets ────────────────────────────────────────────────────────────
print("\n📂 Loading datasets...")
ieee_df = pd.read_csv('data/processed/ieee_processed.csv')
baf_df  = pd.read_csv('data/processed/baf_processed.csv')

def get_Xy(df):
    X = df[FEATURE_NAMES].to_numpy()
    y = df['fraud_label'].to_numpy()
    return X, y

X_ieee, y_ieee = get_Xy(ieee_df)
X_baf,  y_baf  = get_Xy(baf_df)

# 80/20 split — same seed as training
_, X_ieee_test, _, y_ieee_test = train_test_split(X_ieee, y_ieee, test_size=0.2, random_state=42, stratify=y_ieee)
_, X_baf_test,  _, y_baf_test  = train_test_split(X_baf,  y_baf,  test_size=0.2, random_state=42, stratify=y_baf)

print(f"   IEEE test set : {len(y_ieee_test):,} rows  |  fraud: {y_ieee_test.sum():,} ({y_ieee_test.mean()*100:.2f}%)")
print(f"   BAF  test set : {len(y_baf_test):,}  rows  |  fraud: {y_baf_test.sum():,} ({y_baf_test.mean()*100:.2f}%)")

# ── Within-bank Predictions (inflated) ───────────────────────────────────────
print("\n🔮 Running within-bank predictions (inflated - model saw this data)...")
prob_ieee = model.predict(X_ieee_test, batch_size=2048, verbose=1).flatten()
prob_baf  = model.predict(X_baf_test,  batch_size=2048, verbose=1).flatten()

THRESHOLD = 0.5
pred_ieee = (prob_ieee >= THRESHOLD).astype(int)
pred_baf  = (prob_baf  >= THRESHOLD).astype(int)

# ── Cross-bank Predictions (honest generalization) ────────────────────────────
print("\n🔮 Running cross-bank predictions (honest generalization test)...")
prob_ieee_on_baf = model.predict(X_baf, batch_size=2048, verbose=1).flatten()
pred_ieee_on_baf = (prob_ieee_on_baf >= THRESHOLD).astype(int)
prob_baf_on_ieee = model.predict(X_ieee, batch_size=2048, verbose=1).flatten()
pred_baf_on_ieee = (prob_baf_on_ieee >= THRESHOLD).astype(int)

# Combined (both test sets)
X_combined    = np.vstack([X_ieee_test, X_baf_test])
y_combined    = np.concatenate([y_ieee_test, y_baf_test])
prob_combined = np.concatenate([prob_ieee, prob_baf])
pred_combined = (prob_combined >= THRESHOLD).astype(int)

# ── Helper: classification summary ───────────────────────────────────────────
def summary(y_true, y_pred, y_prob, name, tag=''):
    cm      = confusion_matrix(y_true, y_pred)
    rep     = classification_report(y_true, y_pred, target_names=['Legit', 'Fraud'], digits=4)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ap      = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{'='*55}")
    print(f"  {name}  {tag}")
    print(f"{'='*55}")
    print(rep)
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  Avg Precision (PR-AUC): {ap:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return cm, roc_auc, ap

print("\n--- WITHIN-BANK (inflated - model trained on same data) ---")
cm_ieee, auc_ieee, ap_ieee = summary(y_ieee_test, pred_ieee, prob_ieee, "IEEE Credit Card", "[INFLATED]")
cm_baf,  auc_baf,  ap_baf  = summary(y_baf_test, pred_baf,  prob_baf,  "BAF Bank Account", "[INFLATED]")

print("\n--- CROSS-BANK (honest - different bank's unseen data) ---")
cm_ib, auc_ib, ap_ib = summary(y_baf,  pred_ieee_on_baf, prob_ieee_on_baf, "IEEE model on BAF data", "[HONEST]")
cm_bi, auc_bi, ap_bi = summary(y_ieee, pred_baf_on_ieee, prob_baf_on_ieee, "BAF model on IEEE data", "[HONEST]")

cm_comb, auc_comb, ap_comb = summary(y_combined, pred_combined, prob_combined, "Combined Federated View", "")

# ══════════════════════════════════════════════════════════════════
#  FIGURE 1 — Confusion Matrices
# ══════════════════════════════════════════════════════════════════
apply_dark_style()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(STYLE['bg'])
fig.suptitle('Confusion Matrices — Federated Model', fontsize=16, fontweight='bold', color=STYLE['text'], y=1.02)

for ax, cm, title, total in zip(
    axes,
    [cm_ieee, cm_baf, cm_comb],
    ['IEEE Credit Card', 'BAF Bank Account', 'Combined'],
    [len(y_ieee_test), len(y_baf_test), len(y_combined)]
):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred Legit', 'Pred Fraud'], fontsize=10)
    ax.set_yticklabels(['Actual Legit', 'Actual Fraud'], fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold', color=STYLE['accent'], pad=12)
    for i in range(2):
        for j in range(2):
            color = STYLE['fraud'] if (i == j == 1) else (STYLE['legit'] if i == j else STYLE['flag'])
            ax.text(j, i, f'{cm[i,j]:,}\n({cm[i,j]/total*100:.1f}%)',
                    ha='center', va='center', fontsize=11, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor=STYLE['bg'])
print("\n✅ Saved: results/confusion_matrices.png")

# ══════════════════════════════════════════════════════════════════
#  FIGURE 2 — ROC Curves
# ══════════════════════════════════════════════════════════════════
apply_dark_style()
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(STYLE['bg'])
ax.set_facecolor(STYLE['panel'])

for y_true, y_prob, label, color, auc_val, style in [
    (y_ieee_test, prob_ieee,     'IEEE within-bank [inflated]', STYLE['accent'], auc_ieee, '--'),
    (y_baf_test,  prob_baf,      'BAF within-bank [inflated]',  STYLE['legit'],  auc_baf,  '--'),
    (y_baf,  prob_ieee_on_baf,   'IEEE→BAF cross-bank [honest]', '#ff6eb4',      auc_ib,   '-'),
    (y_ieee, prob_baf_on_ieee,   'BAF→IEEE cross-bank [honest]', STYLE['flag'],  auc_bi,   '-'),
]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, label=f'{label}  (AUC={auc_val:.3f})', linewidth=2.2, color=color, linestyle=style)

ax.plot([0,1],[0,1], 'w--', alpha=0.4, label='Random Classifier')
ax.fill_between([0,1],[0,1],[0,1], alpha=0.05, color='white')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — Federated Model', fontsize=14, fontweight='bold', color=STYLE['text'])
ax.legend(loc='lower right', fontsize=10, facecolor=STYLE['panel'], edgecolor=STYLE['accent'])
ax.grid(True)

plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=150, bbox_inches='tight', facecolor=STYLE['bg'])
print("✅ Saved: results/roc_curves.png")

# ══════════════════════════════════════════════════════════════════
#  FIGURE 3 — Precision-Recall Curves
# ══════════════════════════════════════════════════════════════════
apply_dark_style()
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(STYLE['bg'])
ax.set_facecolor(STYLE['panel'])

for y_true, y_prob, label, color, ap_val in [
    (y_ieee_test, prob_ieee, 'IEEE Credit Card', STYLE['accent'],  ap_ieee),
    (y_baf_test,  prob_baf,  'BAF Bank Account', STYLE['legit'],   ap_baf),
    (y_combined,  prob_combined, 'Combined',     STYLE['flag'],    ap_comb),
]:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ax.plot(rec, prec, label=f'{label}  (AP = {ap_val:.4f})', linewidth=2.5, color=color)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves — Federated Model', fontsize=14, fontweight='bold', color=STYLE['text'])
ax.legend(loc='upper right', fontsize=10, facecolor=STYLE['panel'], edgecolor=STYLE['accent'])
ax.grid(True)

plt.tight_layout()
plt.savefig('results/precision_recall_curves.png', dpi=150, bbox_inches='tight', facecolor=STYLE['bg'])
print("✅ Saved: results/precision_recall_curves.png")

# ══════════════════════════════════════════════════════════════════
#  FIGURE 4 — Summary Bar Chart
# ══════════════════════════════════════════════════════════════════
apply_dark_style()
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(STYLE['bg'])
ax.set_facecolor(STYLE['panel'])

metrics = ['ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1']
ieee_scores = [
    auc_ieee, ap_ieee,
    precision_score(y_ieee_test, pred_ieee, zero_division=0),
    recall_score(y_ieee_test, pred_ieee, zero_division=0),
    f1_score(y_ieee_test, pred_ieee, zero_division=0),
]
baf_scores = [
    auc_baf, ap_baf,
    precision_score(y_baf_test, pred_baf, zero_division=0),
    recall_score(y_baf_test, pred_baf, zero_division=0),
    f1_score(y_baf_test, pred_baf, zero_division=0),
]

x = np.arange(len(metrics))
w = 0.35
bars1 = ax.bar(x - w/2, ieee_scores, w, label='IEEE Credit Card', color=STYLE['accent'], alpha=0.85)
bars2 = ax.bar(x + w/2, baf_scores,  w, label='BAF Bank Account', color=STYLE['legit'],  alpha=0.85)

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color=STYLE['text'])

ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Summary — Both Datasets', fontsize=14, fontweight='bold', color=STYLE['text'])
ax.legend(fontsize=11, facecolor=STYLE['panel'], edgecolor=STYLE['accent'])
ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig('results/performance_summary.png', dpi=150, bbox_inches='tight', facecolor=STYLE['bg'])
print("✅ Saved: results/performance_summary.png")

# ══════════════════════════════════════════════════════════════════
#  FIGURE 5 — Fraud Score Distribution
# ══════════════════════════════════════════════════════════════════
apply_dark_style()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(STYLE['bg'])
fig.suptitle('Fraud Probability Distribution', fontsize=15, fontweight='bold', color=STYLE['text'])

for ax, y_true, y_prob, title in [
    (axes[0], y_ieee_test, prob_ieee, 'IEEE Credit Card'),
    (axes[1], y_baf_test,  prob_baf,  'BAF Bank Account'),
]:
    ax.set_facecolor(STYLE['panel'])
    ax.hist(y_prob[y_true == 0], bins=60, alpha=0.7, color=STYLE['legit'], label='Legit', density=True)
    ax.hist(y_prob[y_true == 1], bins=60, alpha=0.7, color=STYLE['fraud'], label='Fraud', density=True)
    ax.axvline(x=0.85, color='white', linestyle='--', linewidth=1.5, label='BLOCK threshold (0.85)')
    ax.axvline(x=0.70, color=STYLE['flag'], linestyle='--', linewidth=1.2, label='FLAG threshold (0.70)')
    ax.set_title(title, fontsize=13, fontweight='bold', color=STYLE['accent'])
    ax.set_xlabel('Fraud Probability', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9, facecolor=STYLE['panel'], edgecolor=STYLE['accent'])
    ax.grid(True)

plt.tight_layout()
plt.savefig('results/score_distribution.png', dpi=150, bbox_inches='tight', facecolor=STYLE['bg'])
print("✅ Saved: results/score_distribution.png")

print("\n" + "="*60)
print("  EVALUATION COMPLETE")
print("="*60)
print(f"\n  [INFLATED - within-bank, model trained on same data]")
print(f"  IEEE within-bank →  ROC-AUC: {auc_ieee:.4f}  |  PR-AUC: {ap_ieee:.4f}")
print(f"  BAF  within-bank →  ROC-AUC: {auc_baf:.4f}  |  PR-AUC: {ap_baf:.4f}")
print(f"\n  [HONEST - cross-bank generalization]")
print(f"  IEEE model on BAF  →  ROC-AUC: {auc_ib:.4f}  |  PR-AUC: {ap_ib:.4f}")
print(f"  BAF  model on IEEE →  ROC-AUC: {auc_bi:.4f}  |  PR-AUC: {ap_bi:.4f}")
print(f"\n  All charts saved to: results/")
print("  Files: confusion_matrices.png | roc_curves.png")
print("         precision_recall_curves.png | performance_summary.png")
print("         score_distribution.png\n")

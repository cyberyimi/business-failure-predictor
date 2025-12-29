"""
Business Failure Predictor - ML Model Training
Predicts company bankruptcy risk using financial metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Neon Orange color scheme for Project 3
NEON_ORANGE = '#ff6600'

print("=" * 70)
print("BUSINESS FAILURE PREDICTOR - MODEL TRAINING")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
df = pd.read_csv('data/american_bankruptcy.csv')

print(f"✅ Loaded {len(df)} records")
print(f"   Companies: {df['company_name'].nunique()}")
print(f"   Years: {df['year'].min()} - {df['year'].max()}")
print(f"   Failed: {len(df[df['status_label']=='failed'])} ({len(df[df['status_label']=='failed'])/len(df)*100:.1f}%)")

# ============================================================================
# 2. PREPARE DATA
# ============================================================================
print("\n🔧 Preparing data for modeling...")

# Convert target to binary
df['failed'] = (df['status_label'] == 'failed').astype(int)

# Select features
feature_cols = [f'X{i}' for i in range(1, 19)]
X = df[feature_cols]
y = df['failed']

print(f"✅ Features: {len(feature_cols)}")
print(f"   Class balance: {y.value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Split data:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled")

# ============================================================================
# 3. TRAIN MODELS
# ============================================================================
print("\n🤖 Training models...")

# Model 1: Random Forest
print("\n   Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_auc = roc_auc_score(y_test, rf_prob)
print(f"   ✅ Random Forest AUC: {rf_auc:.4f}")

# Model 2: Gradient Boosting
print("\n   Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_prob = gb_model.predict_proba(X_test_scaled)[:, 1]
gb_auc = roc_auc_score(y_test, gb_prob)
print(f"   ✅ Gradient Boosting AUC: {gb_auc:.4f}")

# Select best model
if gb_auc > rf_auc:
    best_model = gb_model
    best_pred = gb_pred
    best_prob = gb_prob
    best_name = "Gradient Boosting"
    print(f"\n🏆 Best Model: Gradient Boosting (AUC: {gb_auc:.4f})")
else:
    best_model = rf_model
    best_pred = rf_pred
    best_prob = rf_prob
    best_name = "Random Forest"
    print(f"\n🏆 Best Model: Random Forest (AUC: {rf_auc:.4f})")

# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['Alive', 'Failed']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

if best_name == "Random Forest":
    importances = rf_model.feature_importances_
else:
    importances = gb_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n📊 Creating visualizations...")

plt.style.use('dark_background')

# Visualization 1: Feature Importance
fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance_df.head(15)
bars = ax.barh(range(len(top_features)), top_features['importance'], 
               color=NEON_ORANGE, edgecolor='white', linewidth=2)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=14, fontweight='bold')
ax.set_title('Top 15 Features for Predicting Business Failure', 
             fontsize=16, fontweight='bold', color=NEON_ORANGE, pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.close()
print("✅ Saved: feature_importance.png")

# Visualization 2: Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap=[[0, 0, 0], [1, 0.4, 0]], 
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='white',
            xticklabels=['Predicted Alive', 'Predicted Failed'],
            yticklabels=['Actually Alive', 'Actually Failed'],
            annot_kws={'size': 16, 'weight': 'bold'})
ax.set_title('Confusion Matrix', fontsize=18, fontweight='bold', 
             color=NEON_ORANGE, pad=20)
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.close()
print("✅ Saved: confusion_matrix.png")

# Visualization 3: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_prob)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, color=NEON_ORANGE, linewidth=3, label=f'ROC Curve (AUC = {roc_auc_score(y_test, best_prob):.4f})')
ax.plot([0, 1], [0, 1], 'w--', linewidth=2, label='Random Guess')
ax.fill_between(fpr, tpr, alpha=0.3, color=NEON_ORANGE)
ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('ROC Curve - Business Failure Prediction', 
             fontsize=16, fontweight='bold', color=NEON_ORANGE, pad=20)
ax.legend(loc='lower right', fontsize=12)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('visualizations/roc_curve.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.close()
print("✅ Saved: roc_curve.png")

# Visualization 4: Prediction Distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(best_prob[y_test == 0], bins=50, alpha=0.7, color='#00ff00', 
        label='Actually Alive', edgecolor='white', linewidth=1)
ax.hist(best_prob[y_test == 1], bins=50, alpha=0.7, color=NEON_ORANGE, 
        label='Actually Failed', edgecolor='white', linewidth=1)
ax.set_xlabel('Predicted Failure Probability', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Failure Risk Scores', 
             fontsize=16, fontweight='bold', color=NEON_ORANGE, pad=20)
ax.legend(fontsize=12)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('visualizations/risk_distribution.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.close()
print("✅ Saved: risk_distribution.png")

# ============================================================================
# 7. SAVE MODEL
# ============================================================================
print("\n💾 Saving model and scaler...")

with open('model/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save model metadata
metadata = {
    'model_type': best_name,
    'auc_score': roc_auc_score(y_test, best_prob),
    'features': feature_cols,
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

with open('model/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("✅ Saved model, scaler, and metadata")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\n✅ Model: {best_name}")
print(f"✅ AUC Score: {roc_auc_score(y_test, best_prob):.4f}")
print(f"✅ Accuracy: {(best_pred == y_test).mean():.4f}")
print(f"✅ 4 visualizations created")
print(f"✅ Model saved and ready for predictions")

print("\n🚀 Ready to build the web interface!")

import xgboost as xgb
import optuna
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import base
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*delayed.*Parallel.*"
)

# soll optuna die Hyperparameter erneut optimieren
calculate_hyperparams = False

# train_df und test_df
train_df, test_df = base.load_and_split_data()

# Label-Encoding für die Zielspalte
le = LabelEncoder()
train_df['label_encoded'] = le.fit_transform(train_df['label'])
test_df['label_encoded'] = le.transform(test_df['label'])

# TF-IDF-Vektorisierung des Textes
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_text = tfidf.fit_transform(train_df['text_'])
X_test_text = tfidf.transform(test_df['text_'])

# Numerische Features
X_train_num = train_df[['rating']].values
X_test_num = test_df[['rating']].values

# Kombiniere numerische und Text-Features
X_train = hstack([X_train_num, X_train_text])
X_test = hstack([X_test_num, X_test_text])

y_train = train_df['label_encoded']
y_test = test_df['label_encoded']


# Optuna Objective Function
def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }

    # Split für Cross-Validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(**param)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc


if calculate_hyperparams:
    # Hyperparameter-Tuning
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    # Berechnete beste Hyperparameter
    best_params = study.best_params
else:
    # Manuell gesetzte Hyperparameter (aus früherer Optimierung mit Optuna)
    best_params = {'max_depth': 10,
                   'learning_rate': 0.2966627365806858,
                   'subsample': 0.9375892407928443,
                   'colsample_bytree': 0.8169424111982259,
                   'gamma': 3.655134647675474,
                   'lambda': 0.1407520017524045,
                   'alpha': 0.36057611649921273,
                   'min_child_weight': 6}
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist'
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

# ============================================================
# LEARNING CURVE (XGBOOST)
# ============================================================

train_sizes, train_scores, val_scores = learning_curve(
    estimator=xgb.XGBClassifier(**best_params),
    X=X_train,
    y=y_train,
    cv=3,
    scoring="accuracy",
    train_sizes=np.linspace(0.3, 1.0, 5),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)

val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(8, 5))

plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
plt.plot(train_sizes, val_mean, marker="o", label="Validation Accuracy")

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2
)

plt.fill_between(
    train_sizes,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.2
)

plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("XGBoost Learning Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(base.figures_dir, "class_xgboost_learning_curve.png")
)

plt.close()

# Evaluation auf Testdaten
y_pred = final_model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("Classification Report:\n", report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("XGBoost - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    os.path.join(
        base.figures_dir,
        "class_xgboost_confusion_matrix.png"
    )
)

plt.close()

# Falsch und korrekt klassifizierte Datensätze je in ein File schreiben
correct_mask = (y_pred == y_test.values)
wrong_mask = ~correct_mask
correct_df = test_df.loc[correct_mask].sort_values("id")
wrong_df = test_df.loc[wrong_mask].sort_values("id")
base.reinitialize_folders(base.predictions_dir, drop_existing=False)
correct_df.to_csv(os.path.join(base.predictions_dir, "correct_predictions_xgboost.csv"), index=False)
wrong_df.to_csv(os.path.join(base.predictions_dir, "wrong_predictions_xgboost.csv"), index=False)
print("Path to predictions:", base.predictions_dir)

# Top Features ausgeben
text_features = tfidf.get_feature_names_out()
num_features = ["rating"]
all_features = list(num_features) + list(text_features)
booster = final_model.get_booster()
importance = booster.get_score(importance_type="gain")
importance_df = pd.DataFrame({
    "feature": [all_features[int(k[1:])] for k in importance.keys()],
    "importance": list(importance.values())
})
importance_df = importance_df.sort_values("importance", ascending=False)
print("\nTop Features:")
print(importance_df.head(20))
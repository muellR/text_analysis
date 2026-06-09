from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import base
import optuna
import os
import numpy as np
import pandas as pd

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
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  # 'auto' entfernt
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    # Split für Cross-Validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = RandomForestClassifier(**param, n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
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
    best_params = {
        'n_estimators': 705,
        'max_depth': 38,
        'min_samples_split': 6,
        'min_samples_leaf': 1,
        'max_features': 'log2',
        'bootstrap': True
    }

final_model = RandomForestClassifier(**best_params, n_jobs=-1, random_state=42)
final_model.fit(X_train, y_train)

# Evaluation auf Testdaten
y_pred = final_model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("Classification Report:\n", report)

# Falsch und korrekt klassifizierte Datensätze je in ein File schreiben
correct_mask = (y_pred == y_test.values)
wrong_mask = ~correct_mask
correct_df = test_df.loc[correct_mask].sort_values("id")
wrong_df = test_df.loc[wrong_mask].sort_values("id")
base.reinitialize_folders(base.predictions_dir, drop_existing=False)
correct_df.to_csv(os.path.join(base.predictions_dir, "correct_predictions_random_forest.csv"), index=False)
wrong_df.to_csv(os.path.join(base.predictions_dir, "wrong_predictions_random_forest.csv"), index=False)
print("Path to predictions:", base.predictions_dir)

# Top Features ausgeben
text_features = tfidf.get_feature_names_out()
num_features = ["rating"]
all_features = np.array(list(num_features) + list(text_features))
importance = final_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": importance
})
importance_df = importance_df.sort_values("importance", ascending=False)
print("\nTop Features:")
print(importance_df.head(20))
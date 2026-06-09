# analyze2.py „Was passiert?“

import os
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2
from statsmodels.stats.contingency_tables import mcnemar

import base




# ============================================================
# LOAD DATA
# ============================================================

file_path = os.path.join(base.predictions_dir, "combined_predictions.csv")

df = pd.read_csv(file_path)

MODELS = [
    "class_fcnn_bge",
    "random_forest",
    "xgboost"
]

print(f"Records: {len(df):,}")


# ============================================================
# 1. OVERALL PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("1. OVERALL PERFORMANCE")
print("=" * 80)

overall = pd.DataFrame({
    "accuracy": [df[m].mean() for m in MODELS]
}, index=MODELS)

overall = overall.sort_values("accuracy", ascending=False)

print(overall)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    x=overall.index,
    y=overall["accuracy"],
    ax=ax
)

for bar in ax.patches:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{bar.get_height() * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

ax.set_title("Overall Accuracy")
ax.set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(base.figures_dir, "all_accuracy.png"))
plt.show(block=False)



# ============================================================
# 2. MCNEMAR TESTS
# ============================================================

print("\n" + "=" * 80)
print("2. MCNEMAR TESTS")
print("=" * 80)

ALPHA = 0.05

for m1, m2 in combinations(MODELS, 2):

    both_correct = ((df[m1] == 1) & (df[m2] == 1)).sum()
    m1_only = ((df[m1] == 1) & (df[m2] == 0)).sum()
    m2_only = ((df[m1] == 0) & (df[m2] == 1)).sum()
    both_wrong = ((df[m1] == 0) & (df[m2] == 0)).sum()

    table = [
        [both_correct, m1_only],
        [m2_only, both_wrong]
    ]

    result = mcnemar(table, exact=False, correction=True)

    p = result.pvalue
    significant = "SIGNIFICANT" if p < ALPHA else "not significant"

    print(f"\n{m1} vs {m2}")
    print(f"p-value = {result.pvalue:.6f}")
    print(f"Significance: {significant}")


# ============================================================
# 3. CATEGORY ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("3. CATEGORY ANALYSIS")
print("=" * 80)

category_perf = (
    df.groupby("category")[MODELS]
    .mean()
    .sort_index()
)

print(category_perf.head())

category_perf.to_csv(
    os.path.join(base.figures_dir, "all_category_performance.csv")
)

plt.figure(figsize=(12, 8))
sns.heatmap(
    category_perf,
    annot=True,
    cmap="viridis",
    fmt=".2f"
)
plt.title("Accuracy by Category")
plt.tight_layout()
plt.savefig(os.path.join(base.figures_dir, "all_category_performance_heatmap.png"))
plt.show(block=False)


# ============================================================
# 4. RATING ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("4. RATING ANALYSIS")
print("=" * 80)

rating_perf = (
    df.groupby("rating")[MODELS]
    .mean()
)

print(rating_perf)

plt.figure(figsize=(10, 6))

for m in MODELS:
    plt.plot(
        rating_perf.index,
        rating_perf[m],
        marker="o",
        label=m
    )

plt.legend()
plt.title("Accuracy by Rating")
plt.xlabel("Rating")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base.figures_dir, "all_rating_performance.png"))
plt.show(block=False)


# ============================================================
# 5. MODEL AGREEMENT
# ============================================================

print("\n" + "=" * 80)
print("5. MODEL AGREEMENT")
print("=" * 80)

agreement = (
    df[MODELS]
    .astype(str)
    .agg("-".join, axis=1)
    .value_counts()
)

print(agreement)

agreement.to_csv(
    os.path.join(base.figures_dir, "all_model_agreement.csv")
)


# ============================================================
# 6. ENSEMBLE
# ============================================================

print("\n" + "=" * 80)
print("6. ENSEMBLE")
print("=" * 80)

df["ensemble_vote"] = (
    df[MODELS].sum(axis=1) >= 2
).astype(int)

ensemble_acc = df["ensemble_vote"].mean()

print(f"Ensemble Accuracy: {ensemble_acc:.4f}")

comparison = overall.copy()
comparison.loc["ensemble"] = ensemble_acc

comparison = comparison.sort_values(
    "accuracy",
    ascending=False
)

print(comparison)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    x=comparison.index,
    y=comparison["accuracy"],
    ax=ax
)

for bar in ax.patches:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{bar.get_height() * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

ax.set_title("Model vs Ensemble")
plt.tight_layout()
plt.savefig(os.path.join(base.figures_dir, "all_model_vs_ensemble.png"))
plt.show(block=False)


# ============================================================
# 7. ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("7. ERROR ANALYSIS")
print("=" * 80)

all_wrong = df[
    (df["class_fcnn_bge"] == 0)
    & (df["random_forest"] == 0)
    & (df["xgboost"] == 0)
]

print(f"All models wrong: {len(all_wrong)}")
print(f"distinct categories: {df['category'].nunique()}")

all_wrong.to_csv(
    os.path.join(base.figures_dir, "all_models_wrong.csv"),
    index=False
)

print("\nTop categories among all-fail cases:")
print(
    all_wrong["category"]
    .value_counts()
    #.head(20)
)




# ============================================================
# 8. TEXT LENGTH ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("8. TEXT LENGTH ANALYSIS")
print("=" * 80)

df["char_length"] = (
    df["text_"]
    .fillna("")
    .astype(str)
    .str.len()
)

df["word_length"] = (
    df["text_"]
    .fillna("")
    .astype(str)
    .str.split()
    .str.len()
)

df["word_cnt_bin"] = pd.qcut(
    df["word_length"],
    q=5,
    duplicates="drop"
)

# lesbare Labels
df["word_cnt_bin"] = df["word_cnt_bin"].apply(
    lambda b: f"{int(b.left)+1}–{int(b.right)} Wörter"
)

length_perf = (
    df.groupby("word_cnt_bin")[MODELS]
    .mean()
)

print(length_perf)

# Heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    length_perf * 100,
    annot=True,
    fmt=".1f",
    cmap="YlGn",
    ax=ax
)
ax.set_title("Accuracy by Text Length")
ax.set_ylabel("Word Count Bin")
ax.set_xlabel("Model")
plt.tight_layout()
plt.savefig(os.path.join(base.figures_dir, "all_length_performance.png"))
plt.show(block=False)







# ============================================================
# 9. DIFFICULTY INDEX
# ============================================================

print("\n" + "=" * 80)
print("9. DIFFICULTY INDEX")
print("=" * 80)

df["difficulty"] = 3 - df[MODELS].sum(axis=1)

difficulty_counts = (
    df["difficulty"]
    .value_counts()
    .sort_index()
)

difficulty_pct = (difficulty_counts / difficulty_counts.sum() * 100).round(1)

difficulty_summary = pd.DataFrame({
    "count": difficulty_counts,
    "pct": difficulty_pct.astype(str) + "%"
})

print(difficulty_summary)


# Durchschnittliche Schwierigkeit pro Kategorie
# in sovieln Modellen pro 3 Modellen falsch
difficulty_category = (
    df.groupby("category")["difficulty"]
    .mean()
    .sort_values(ascending=False)
)

print("\nMost difficult categories:")
print(difficulty_category.head(20))


# ============================================================
# 10. HARDEST CASES EXPORT
# ============================================================

print("\n" + "=" * 80)
print("10. HARDEST CASES")
print("=" * 80)

hardest = (
    df.sort_values(
        ["difficulty", "word_length"],
        ascending=[False, False]
    )
)

hardest_50 = hardest.head(50)

hardest_50.to_csv(
    os.path.join(base.figures_dir, "all_hardest_50_cases.csv"),
    index=False
)

print(hardest_50[
    [
        "id",
        "category",
        "rating",
        "difficulty",
        "word_length"
    ]
].head(20))
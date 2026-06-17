# analyze4.py – „Welche Fakes entkommen den Modellen?"
# Ziel: Charakterisierung von CG-Reviews, die von ≥2 Modellen NICHT erkannt wurden.
# Diese sind die gefährlichsten, weil sie systematisch durchschlüpfen.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import base


# ============================================================
# SETUP
# ============================================================
print("\n" + "=" * 80 + "\n0. LOAD DATA\n" + "=" * 80)

file_path = os.path.join(base.predictions_dir, "combined_predictions.csv")
df = pd.read_csv(file_path)

MODELS = ["class_fcnn_bge", "random_forest", "xgboost"]

# Anzahl Modelle, die RICHTIG lagen (bei CG: korrekt = predicted als fake)
# Achtung: Die Modell-Spalten enthalten 1 = korrekt klassifiziert, 0 = Fehler
# (basierend auf analyze2.py: df[m].mean() = accuracy)
df["correct_count"] = df[MODELS].sum(axis=1)
df["difficulty"] = 3 - df["correct_count"]   # 0 = alle richtig, 3 = alle falsch

# Gruppen
OR = df[df["label"] == "OR"].copy()                              # Echte Reviews
CG_all = df[df["label"] == "CG"].copy()                         # Alle Fakes
CG_caught = CG_all[CG_all["difficulty"] <= 1].copy()            # ≤1 Modell falsch → erkannt
CG_escaped = CG_all[CG_all["difficulty"] >= 2].copy()           # ≥2 Modelle falsch → entkommen

print(f"Echte Reviews (OR):          {len(OR):>6,}")
print(f"Alle Fakes (CG):             {len(CG_all):>6,}")
print(f"  davon erkannt (diff ≤ 1):  {len(CG_caught):>6,}  ({len(CG_caught)/len(CG_all)*100:.1f}%)")
print(f"  davon entkommen (diff ≥ 2):{len(CG_escaped):>6,}  ({len(CG_escaped)/len(CG_all)*100:.1f}%)")
print(f"     diff == 2: {(CG_all['difficulty']==2).sum()}")
print(f"     diff == 3: {(CG_all['difficulty']==3).sum()}")


# ============================================================
# 1. FEATURES BERECHNEN (gecacht)
# ============================================================
print("\n" + "=" * 80 + "\n1. FEATURE-ENGINEERING\n" + "=" * 80)

nlp = spacy.load("en_core_web_lg")

cache_path = os.path.join(base.figures_dir, "14_fn_features.csv")

if os.path.exists(cache_path):
    df = pd.read_csv(cache_path)
    print("Features aus Cache geladen.")
else:
    df["text_"] = df["text_"].fillna("").astype(str)

    # --- Basis ---
    df["char_len"]   = df["text_"].str.len()
    df["word_len"]   = df["text_"].str.split().str.len()
    df["avg_word_len"] = df["text_"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )

    # --- Satzstruktur ---
    def spacy_features(text):
        doc = nlp(text)
        sents = list(doc.sents)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        words  = [t for t in tokens if t.is_alpha]
        adj    = [t for t in words if t.pos_ == "ADJ"]
        nouns  = [t for t in words if t.pos_ == "NOUN"]
        return {
            "sentence_count": len(sents),
            "entity_count":   len(doc.ents),
            "adj_ratio":      len(adj) / max(len(words), 1),
            "noun_ratio":     len(nouns) / max(len(words), 1),
            "adj_noun_ratio": len(adj) / max(len(nouns), 1),
            # Lexikalische Diversität (TTR): einzigartige / alle Wörter
            "ttr":            len(set(w.lower_ for w in words)) / max(len(words), 1),
        }

    features_spacy = df["text_"].apply(spacy_features).apply(pd.Series)
    df = pd.concat([df, features_spacy], axis=1)

    # --- Interpunktion ---
    df["exclamation_count"] = df["text_"].str.count("!")
    df["question_count"]    = df["text_"].str.count(r"\?")
    df["uppercase_ratio"]   = df["text_"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )

    # --- Superlative (Proxy für Überschwänglichkeit) ---
    superlatives = [
        "best", "worst", "greatest", "most", "amazing", "perfect",
        "incredible", "fantastic", "excellent", "awful", "terrible"
    ]
    pattern = r"\b(" + "|".join(superlatives) + r")\b"
    df["superlative_count"] = df["text_"].str.lower().str.count(pattern)
    df["superlative_density"] = df["superlative_count"] / df["word_len"].replace(0, np.nan)

    # --- Specificity (Entity-Dichte) ---
    df["entity_density"] = df["entity_count"] / df["word_len"].replace(0, np.nan)

    # --- Sentiment ---
    df["sentiment"] = df["text_"].apply(lambda x: TextBlob(x).sentiment.polarity)

    df.to_csv(cache_path, index=False)
    print("Features berechnet und gespeichert.")

# Gruppen neu zuweisen nach reload
df["correct_count"] = df[MODELS].sum(axis=1)
df["difficulty"]    = 3 - df["correct_count"]
OR        = df[df["label"] == "OR"].copy()
CG_all    = df[df["label"] == "CG"].copy()
CG_caught  = CG_all[CG_all["difficulty"] <= 1].copy()
CG_escaped = CG_all[CG_all["difficulty"] >= 2].copy()


# ============================================================
# 2. PROFIL-VERGLEICH: Entkommen vs. Erkannt vs. Echt
# ============================================================
print("\n" + "=" * 80 + "\n2. PROFIL-VERGLEICH\n" + "=" * 80)

FEATURES = [
    "word_len", "sentence_count", "avg_word_len",
    "entity_count", "entity_density",
    "adj_ratio", "adj_noun_ratio", "ttr",
    "superlative_density",
    "exclamation_count", "uppercase_ratio",
    "sentiment"
]

profile = pd.DataFrame({
    "echt":                OR[FEATURES].mean().round(2),
    "Fake erkannt":        CG_caught[FEATURES].mean().round(2),
    "Fake entkommen":      CG_escaped[FEATURES].mean().round(2),
})

print(profile.round(4).to_string())


# ============================================================
# 3. HEATMAP: Relative Abweichung vom Mittelwert
# ============================================================
if 0:
    print("\n" + "=" * 80 + "\n3. HEATMAP RELATIVE ABWEICHUNG\n" + "=" * 80)

    #Normiere auf Gesamtmittelwert (z-score über Gruppen)
    profile_T = profile.T
    z = (profile_T - profile_T.mean()) / (profile_T.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        z,
        annot=profile_T.round(3),
        fmt="",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5
    )
    ax.set_title("Profil-Vergleich: Relative Abweichung")
    ax.set_xlabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(base.figures_dir, "14_fn_profile_heatmap.png"), dpi=150)
    plt.close()
    print("→ Gespeichert: 14_fn_profile_heatmap.png")


# ============================================================
# 4. RATING-VERTEILUNG
# ============================================================
if 0:
    print("\n" + "=" * 80 + "\n4. RATING-VERTEILUNG\n" + "=" * 80)

    rating_dist = pd.DataFrame({
        "OR (echt)":    OR["rating"].value_counts(normalize=True).sort_index(),
        "CG erkannt":   CG_caught["rating"].value_counts(normalize=True).sort_index(),
        "CG entkommen": CG_escaped["rating"].value_counts(normalize=True).sort_index(),
    }).fillna(0)

    print(rating_dist.round(3))

    ax = rating_dist.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Rating-Verteilung nach Gruppe")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Anteil")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(base.figures_dir, "14_fn_rating_distribution.png"), dpi=150)
    plt.close()
    print("→ Gespeichert: 14_fn_rating_distribution.png")


# ============================================================
# 5. SENTIMENT × RATING HEATMAP (nur CG entkommen vs. erkannt)
# ============================================================
if 0:
    print("\n" + "=" * 80 + "\n5. SENTIMENT × RATING HEATMAP\n" + "=" * 80)

    def sentiment_rating_heatmap(data, title, filename):
        pivot = data.groupby("rating")["sentiment"].mean().to_frame()
        pivot.columns = ["mean_sentiment"]
        print(f"\n{title}:")
        print(pivot.round(3))
        pivot.plot(kind="bar", legend=False, figsize=(7, 4))
        plt.title(f"Ø Sentiment pro Rating – {title}")
        plt.ylabel("Sentiment (TextBlob polarity)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(base.figures_dir, filename), dpi=150)
        plt.close()

    sentiment_rating_heatmap(CG_escaped, "CG entkommen", "15_fn_sentiment_rating_escaped.png")
    sentiment_rating_heatmap(CG_caught,  "CG erkannt",   "15_fn_sentiment_rating_caught.png")
    print("→ Gespeichert: 15_fn_sentiment_rating_*.png")


# ============================================================
# 6. TF-IDF: Charakteristische Wörter der entkommenen Fakes
# ============================================================
if 0:
    print("\n" + "=" * 80 + "\n6. TF-IDF KEYWORD-PROFIL\n" + "=" * 80)

    # TF-IDF: CG_escaped vs CG_caught
    corpus = pd.concat([
        CG_escaped[["text_"]].assign(group="escaped"),
        CG_caught[["text_"]].assign(group="caught")
    ])

    vec = TfidfVectorizer(stop_words="english", max_features=500, ngram_range=(1, 2))
    X = vec.fit_transform(corpus["text_"])
    feature_names = vec.get_feature_names_out()

    escaped_mask = corpus["group"].values == "escaped"
    caught_mask  = ~escaped_mask

    mean_escaped = X[escaped_mask].mean(axis=0).A1
    mean_caught  = X[caught_mask].mean(axis=0).A1

    tfidf_diff = pd.DataFrame({
        "term":    feature_names,
        "escaped": mean_escaped,
        "caught":  mean_caught,
        "diff":    mean_escaped - mean_caught
    }).sort_values("diff", ascending=False)

    print("\nTop 10 Wörter: überrepräsentiert in ENTKOMMENEN Fakes:")
    print(tfidf_diff.head(10).to_string(index=False))

    print("\nTop 10 Wörter: überrepräsentiert in ERKANNTEN Fakes:")
    print(tfidf_diff.tail(10).sort_values("diff").to_string(index=False))

    #tfidf_diff.to_csv(os.path.join(base.figures_dir, "06_fn_tfidf_diff.csv"), index=False)

    # Barplot Top-10 vs Bottom-10
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    top10 = tfidf_diff.head(10)
    bot10 = tfidf_diff.tail(10).sort_values("diff")

    axes[0].barh(top10["term"], top10["diff"], color="#e74c3c")
    axes[0].set_title("Überrepräsentiert: entkommen")
    axes[0].invert_yaxis()

    axes[1].barh(bot10["term"], bot10["diff"].abs(), color="#2ecc71")
    axes[1].set_title("Überrepräsentiert: erkannt")
    axes[1].invert_yaxis()

    plt.suptitle("TF-IDF Differenz: Entkommene vs. Erkannte CG-Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(base.figures_dir, "16_fn_tfidf_keywords.png"), dpi=150)
    plt.close()
    print("→ Gespeichert: 16_fn_tfidf_keywords.png")


# ============================================================
# 7. LÄNGEN-SWEETSPOT: Ab welcher Länge entkommen mehr Fakes?
# ============================================================
if 0:
    print("\n" + "=" * 80 + "\n7. LÄNGEN-SWEETSPOT\n" + "=" * 80)

    # Für alle CG-Reviews: Escape-Rate pro Längenbin
    CG_all_feat = df[df["label"] == "CG"].copy()
    CG_all_feat["escaped"] = (CG_all_feat["difficulty"] >= 2).astype(int)

    CG_all_feat["word_len_bin"] = pd.cut(
        CG_all_feat["word_len"],
        bins=[0, 20, 50, 100, 200, 500, 9999],
        labels=["1–20", "21–50", "51–100", "101–200", "201–500", "500+"]
    )

    escape_by_len = (
        CG_all_feat.groupby("word_len_bin")["escaped"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "escape_rate", "count": "n"})
    )

    print(escape_by_len)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(escape_by_len.index, escape_by_len["escape_rate"], color="#e67e22")
    ax.set_title("Escape-Rate (≥2 Modelle falsch) nach Textlänge – nur CG")
    ax.set_xlabel("Wortanzahl")
    ax.set_ylabel("Anteil entkommener Fakes")
    for i, (rate, n) in enumerate(zip(escape_by_len["escape_rate"], escape_by_len["n"])):
        ax.text(i, rate + 0.005, f"n={n}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(base.figures_dir, "17_fn_escape_rate_by_length.png"), dpi=150)
    plt.close()
    print("→ Gespeichert: 17_fn_escape_rate_by_length.png")


# ============================================================
# 8. KATEGORIE: Wo entkommen am meisten Fakes?
# ============================================================
if 0:
    print("\n" + "=" * 80 + "\n8. ESCAPE-RATE NACH KATEGORIE\n" + "=" * 80)

    escape_by_cat = (
        CG_all_feat.groupby("category")["escaped"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "escape_rate", "count": "n"})
        .sort_values("escape_rate", ascending=False)
    )

    print("\nTop 15 Kategorien mit höchster Escape-Rate:")
    print(escape_by_cat.head(15).round(3).to_string())

    escape_by_cat.head(20).to_csv(
        os.path.join(base.figures_dir, "18_fn_escape_rate_by_category.csv")
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    top_cats = escape_by_cat.head(20)
    ax.barh(top_cats.index[::-1], top_cats["escape_rate"][::-1], color="#8e44ad")
    ax.set_title("Top-20 Kategorien: Escape-Rate (≥2 Modelle falsch)")
    ax.set_xlabel("Escape-Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(base.figures_dir, "18_fn_escape_rate_category.png"), dpi=150)
    plt.close()
    print("→ Gespeichert: 18_fn_escape_rate_category.png")




# ============================================================
# 10. ZUSAMMENFASSUNG
# ============================================================
print("\n" + "=" * 80 + "\n10. ZUSAMMENFASSUNG\n" + "=" * 80)

print("""
Was zeichnet entkommene CG-Reviews aus?
─────────────────────────────────────────────────────────────────────
Vergleiche die Ausgaben oben mit diesen Hypothesen:

 word_len ↓          → Entkommene Fakes sind halb so lang wie erkannte
 sentence_count ↓    → Wenige Sätze, wenig Struktur
 ttr ↓               → Geringe lexikalische Vielfalt (Wiederholung)
 adj_noun_ratio ↑    → Viele Adjektive auf wenig Nomen (vage Lob-Sprache)   

=============
 entity_count.       Deutlich weniger konkrete Namen, Orte, Produktreferenzen als echte Reviews.
 entity_density ↓    → Wenig konkrete Namen / Orte / Produkte
 superlative_density ↑→ Überschwänglichkeit ("best", "amazing", "perfect")
 sentiment ↑         → Übertrieben positive Tonlage
 
 → Entkommene Fakes klingen wie echte, aber extrem generische Reviews.
─────────────────────────────────────────────────────────────────────
""")

print("Done!")
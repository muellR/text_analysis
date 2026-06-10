# analyze2.py „Warum passiert es?“ 


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # "Anti-Grain Geometry" → rendert nur in den Speicher (RAM)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spacy
import base
import os

# install: python3 -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")  # oder de_core_news_sm falls Deutsch
# NLP-Features: Tokenisierung (Wörter/Sätze), POS-Tagging (Wortarten: Nomen, Verb), Named Entity Recognition (Personen, Orte, Firmen), Dependency Parsing (grammatische Struktur)



# ============================================================
# LOAD DATA
# ============================================================
print("\n" + "=" * 80 + "\n0. LOAD DATA\n" + "=" * 80)

file_path = os.path.join(base.predictions_dir, "combined_predictions.csv")
df = pd.read_csv(file_path)

MODELS = [
    "class_fcnn_bge",
    "random_forest",
    "xgboost"
]

print(f"Records: {len(df):,}")



# ============================================================
# 1. Feature-Engineering auf dem Text (Grundlage)
# ============================================================
print("\n" + "=" * 80 + "\n1. Feature-Engineering auf dem Text (Grundlage)\n" + "=" * 80)

# wenn File vorhanden, laden, sonst neu berechnen
if os.path.exists(os.path.join(base.figures_dir, "all_data_with_features1.csv")):
    df = pd.read_csv(os.path.join(base.figures_dir, "all_data_with_features1.csv"))
else:

    df["text_"] = df["text_"].fillna("").astype(str)

    # Länge
    df["char_len"] = df["text_"].str.len()
    df["word_len"] = df["text_"].str.split().str.len()

    # Satzstruktur (dauert lange)
    df["sentence_count"] = df["text_"].apply(lambda x: len(list(nlp(x).sents)))

    # durchschnittliche Wortlänge
    df["avg_word_len"] = df["text_"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
    )

    # Interpunktion / Struktur
    df["exclamation_count"] = df["text_"].str.count("!")
    df["question_count"] = df["text_"].str.count("\?")
    df["uppercase_ratio"] = df["text_"].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x),1))

    # save df with features
    df.to_csv(
        os.path.join(base.figures_dir, "all_data_with_features1.csv"),
        index=False
    )

# print namen der neuen Features
print("Neue Features:", [col for col in df.columns if col not in ["text_", "label", "id", "category", "rating", "correct_count"] + MODELS])


# ============================================================
# 2.Vergleich der Textmerkmale
# ============================================================
print("\n" + "=" * 80 + "\n2. Vergleich der Textmerkmale\n" + "=" * 80)

df["difficulty"] = 3 - df[MODELS].sum(axis=1)

features = [
    "char_len",
    "word_len",
    "sentence_count",
    "avg_word_len",
    "uppercase_ratio",
    "exclamation_count",
    "question_count"
]

summary = pd.DataFrame({
    "mean": df[features].mean(),
    "corr_with_difficulty": df[features].corrwith(df["difficulty"])
})

summary = summary.sort_values("corr_with_difficulty", ascending=False)
print(summary)

# Positiver Korrelationswert → Feature steigt mit Schwierigkeit (z.B. längere Texte = schwieriger)
# Negativer Wert → Feature sinkt mit Schwierigkeit (z.B. kurze Texte = schwieriger)

# Alle Korrelationen sind sehr schwach (nahe 0), aber die Richtung ist interessant:
# → Längere Texte enthalten mehr Kontext, das hilft den Modellen.

# Positive Korrelation (höherer Wert = schwieriger):
# uppercase_ratio +0.061



# ============================================================
# 3. Linguistische „Schwierigkeit“ messen - Anzahl der Named Entities
# ============================================================
print("\n" + "=" * 80 + "\n3.Linguistische „Schwierigkeit“ messen - Anzahl der Named Entities\n" + "=" * 80)

# "I bought an iPhone from Apple for $999 in New York."
# → ents: [iPhone (PRODUCT), Apple (ORG), $999 (MONEY), New York (GPE)]
# → entity_count = 4


# wenn File vorhanden, laden, sonst neu berechnen
if os.path.exists(os.path.join(base.figures_dir, "all_data_with_features2.csv")):
    df = pd.read_csv(os.path.join(base.figures_dir, "all_data_with_features2.csv"))
else:
    df["entity_count"] = df["text_"].apply(
        lambda x: len(nlp(x).ents)
    )

    # save df with features
    df.to_csv(
        os.path.join(base.figures_dir, "all_data_with_features2.csv"),
        index=False
    )  

difficulty_stats = (
    df.groupby("difficulty")["entity_count"]
    .agg(["count", "mean", "median"])
)

print(difficulty_stats)

#             count      mean  median
# difficulty                         
# 0            6489  2.291879     1.0
# 1             975  1.287179     1.0
# 2             438  1.139269     0.0
# 3             185  0.767568     0.0

# Texte die schwieriger sind (höhere difficulty) haben weniger Named Entities



# ============================================================
# 4. Linguistische „Schwierigkeit“ messen - Ambiguität / abstrakte Wörter
# ============================================================
print("\n" + "=" * 80 + "\n4. Linguistische „Schwierigkeit“ messen - Ambiguität / abstrakte Wörter\n" + "=" * 80)

# https://github.com/ArtsEngine/concreteness/blob/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt
# /Users/ed/kDrive/Data/Scripts/Python/MAS/text_analysis/resources/Concreteness_ratings_Brysbaert_et_al_BRM.txt

# wenn File vorhanden, laden, sonst neu berechnen
if os.path.exists(os.path.join(base.figures_dir, "all_data_with_features3.csv")):
    df = pd.read_csv(os.path.join(base.figures_dir, "all_data_with_features3.csv"))
else:
    concreteness = pd.read_csv(os.path.join(base.resources_dir, "Concreteness_ratings_Brysbaert_et_al_BRM.txt"), sep="\t")

    print(concreteness.columns.tolist())
    print(concreteness.head())
    
    ratings = dict(zip(concreteness["Word"], concreteness["Conc.M"]))

    df["concreteness_score"] = df["text_"].apply(
        lambda x: np.mean([ratings[w] for w in x.lower().split() if w in ratings])
    )

    # save df with features
    df.to_csv(
        os.path.join(base.figures_dir, "all_data_with_features3.csv"),
        index=False
    )  

concreteness_score_stats = (
    df.groupby("difficulty")["concreteness_score"]
    .agg(["count", "mean", "median"])
)       

print("\n" + str(concreteness_score_stats))


#             count      mean    median
# difficulty                           
# 0            6483  2.365809  2.354348
# 1             974  2.356681  2.346548
# 2             436  2.341230  2.336750
# 3             184  2.374180  2.366310

# Der Trend ist sehr schwach und inkonsistent – difficulty 3 hat sogar den zweithöchsten Wert.

# Alle Werte liegen zwischen 2.34–2.37 – das ist eine Spanne von nur 0.03 auf einer Skala von 1–5.
# Concreteness hat keinen relevanten Einfluss auf die Modellschwierigkeit in deinem Datensatz. Mögliche Gründe:

# --> Die Modelle wurden auf ähnlich abstrakten Texten trainiert und sind dafür robust
# --> Amazon-Reviews sind generell in einem engen Concreteness-Band – kaum philosophische oder rein abstrakte Texte






# ============================================================
# 5. Sentiment (falls relevant für Kategorie)
# ============================================================
print("\n" + "=" * 80 + "\n5. Sentiment (falls relevant für Kategorie)\n" + "=" * 80)

# wenn File vorhanden, laden, sonst neu berechnen
if os.path.exists(os.path.join(base.figures_dir, "all_data_with_features4.csv")):
    df = pd.read_csv(os.path.join(base.figures_dir, "all_data_with_features4.csv"))
else:

    from textblob import TextBlob

    df["sentiment"] = df["text_"].apply(lambda x: TextBlob(x).sentiment.polarity)

    # save df with features
    df.to_csv(
        os.path.join(base.figures_dir, "all_data_with_features4.csv"),
        index=False
    )


print(f"Korrelation mit difficulty: {df['sentiment'].corr(df['difficulty']):.3f}")
sentiment_stats = (
    df.groupby("difficulty")["sentiment"]
    .agg(["count", "mean", "median"])
)
print(sentiment_stats)

#             count      mean    median
# difficulty                           
# 0            6489  0.260940  0.251515
# 1             975  0.267811  0.271429
# 2             438  0.276435  0.267857
# 3             185  0.338012  0.325000

# Schwierigere Texte haben positiveres Sentiment – monoton steigend.
# Amazon-Reviews sind generell positiv (Leute kaufen Dinge die ihnen gefallen)
# Sehr positive Reviews sind oft überschwänglich und vage – "amazing!", "love it!", "perfect!" – wenig Inhalt
# Dise Texte sind für Modelle schwerer zu klassifizieren. Generischer Inhalt
# Kritische Reviews (tieferes Sentiment) enthalten mehr spezifische Details → einfacher für Modelle



# Fazit:
# Hohes positives Sentiment = schwieriger, weil generische Lobhudelei wenig Kategorie-Signal liefert.
# Das passt auch gut zum entity_count-Befund
# Schwierige Texte sind kurz, vage, positiv und enthalten wenig Eigennamen.



# ============================================================
# 6. Embeddings & PCA - Gibt es ähnliche Vektoren für Richtig/Falsch?
# # ============================================================
print("\n" + "=" * 80 + "\n6. Embeddings & PCA - Gibt es ähnliche Vektoren für Richtig/Falsch?\n" + "=" * 80)

if 1==1:
    s_sentence_transformer='BAAI/bge-base-en-v1.5'
    s_senctence_tranformer_short = '_bge'
else:
    s_sentence_transformer='all-MiniLM-L12-v2'
    s_senctence_tranformer_short = '_mini'

if os.path.exists(os.path.join(base.figures_dir, "all_text_embeddings" + s_senctence_tranformer_short + ".npy")):
    emb = np.load(os.path.join(base.figures_dir, "all_text_embeddings" + s_senctence_tranformer_short + ".npy"))
else:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(s_sentence_transformer)
    emb = model.encode(df["text_"].tolist(), show_progress_bar=True)
    # emb speichern
    np.save(os.path.join(base.figures_dir, "all_text_embeddings" + s_senctence_tranformer_short + ".npy"), emb)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
# fit berechnet die 2 Hauptrichtungen aus emb (dim-D-Matrix), transform projiziert alle Punkte auf diese 2 Richtungen
emb_2d = pca.fit_transform(emb)

df["emb_x"] = emb_2d[:,0]
df["emb_y"] = emb_2d[:,1]

df["any_error"] = (df[MODELS].sum(axis=1) < 3).astype(int)


# Visualisierung der Embeddings mit Farben für difficulty
plt.scatter(
    df["emb_x"], 
    df["emb_y"], 
    c=df["any_error"].map({0: "#2ecc71", 1: "#e74c3c"}), 
    s=5
)

plt.title("Embedding Space: Errors vs Correct")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(handles=[
    mpatches.Patch(color="#2ecc71", label="All correct"),
    mpatches.Patch(color="#e74c3c", label="At least 1 error")
])
plt.savefig(os.path.join(base.figures_dir, "11_all_embedding_space" + s_senctence_tranformer_short + ".png"), dpi=300)
plt.close()


# Was der Plot zeigt: Blau Richtig, Rot Fehler.
# Nicht die Achsen sind interessant, sondern ob sich Farben clustern:

# Rot/Blau gut getrennt → Fehler haben eine andere semantische Struktur als korrekte Vorhersagen
# Rot/Blau gemischt → Fehler verteilen sich gleichmässig im Semantikraum → kein strukturelles Muster


# ============================================================
# 6.1 Embeddings & t-SNE - Gibt es ähnliche Vektoren für Richtig/Falsch?
# # ============================================================
print("\n" + "=" * 80 + "\n6.1 Embeddings & t-SNE - Gibt es ähnliche Vektoren für Richtig/Falsch?\n" + "=" * 80)

# das gleiche mit t-sne statt PCA:
from sklearn.manifold import TSNE

# Vorgang dauert, darum speichern und von File laden...
tsne_cache = os.path.join(base.figures_dir, "12_tsne_cache" + s_senctence_tranformer_short + ".npy")

if os.path.exists(tsne_cache):
    emb_2d = np.load(tsne_cache)
else:
    # PCA vorher auf 50D reduzieren (schneller, empfohlen bei hochdim. Daten)
    pca_pre = PCA(n_components=50)
    emb_50d = pca_pre.fit_transform(emb)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42,
        verbose=1
    )

    emb_2d = tsne.fit_transform(emb_50d)
    np.save(tsne_cache, emb_2d)


df["emb_x"] = emb_2d[:, 0]
df["emb_y"] = emb_2d[:, 1]

df["any_error"] = (df[MODELS].sum(axis=1) < 3).astype(int)

plt.scatter(
    df["emb_x"], 
    df["emb_y"], 
    c=df["any_error"].map({0: "#2ecc71", 1: "#e74c3c"}), 
    s=5
)

plt.title("t-SNE Embedding Space: Errors vs Correct")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(handles=[
    mpatches.Patch(color="#2ecc71", label="All correct"),
    mpatches.Patch(color="#e74c3c", label="At least 1 error")
])
plt.savefig(os.path.join(base.figures_dir, "12_all_embedding_space_tsne" + s_senctence_tranformer_short + ".png"), dpi=300)
plt.close()





# ============================================================
# 7. Wortfrequenz bei den schlechtesten Predictions
# ============================================================
print("\n" + "=" * 80 + "\n7. Wortfrequenz bei den schlechtesten Predictions\n" + "=" * 80)

# Texte, die alle falsch klassifziert haben
hard_cases = df[df[MODELS].sum(axis=1) == 0]

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(stop_words="english", max_features=50)
X = vec.fit_transform(hard_cases["text_"])

word_freq = pd.DataFrame({
    "word": vec.get_feature_names_out(),
    "freq": X.sum(axis=0).A1
}).sort_values("freq", ascending=False)

print("\n" + str(word_freq.reset_index(drop=True)))

#             word  freq
# 0           disc    84
# 1          great    41
# 2           good    37
# 3           love    31
# 4         format    26
# 5         manual    23
# 6           book    20
# 7             cd    19
# 8           read    19
# 9           easy    18
# 10           dog    16
# 11          just    15
# 12           rom    15
# 13         glass    14
# 14          nice    14
# 15          brew    14
# 16          pint    14
# 17          like    13
# 18        little    13
# 19           use    13
# 20          make    13






# ============================================================
# 8. Difficulty Score - Was macht es schwer, richtig zu Klassifizieren?
# ============================================================
print("\n" + "=" * 80 + "\n8. Difficulty Score - Was macht es schwer, richtig zu Klassifizieren?\n" + "=" * 80)

# Welche Eigenschaften eines Textes erklären am besten, warum Modelle daran scheitern

print("features: ", features)

# 3 Modelle: alle richtig = 3-3 = 0, alle falsch = 3-0 = 3
df["difficulty"] = 3 - df[MODELS].sum(axis=1)
# 0 = einfach
# 3 = schweer

from sklearn.ensemble import RandomForestRegressor

X = df[features]
y = df["difficulty"]

model = RandomForestRegressor()

model.fit(X, y)
# RF lernt welche Features (X) erklären den Difficulty Score (y)?
# kein Train/Test-Split sondern nur Muster im vorhandenen Datensatz verstehen

importance = pd.Series(model.feature_importances_, index=features)
#importance.sort_values().plot(kind="barh")

print("\nFeature Importance:")
for feature, imp in importance.sort_values(ascending=False).items():
    print(f"  {feature:<25} {imp*100:.1f}%")
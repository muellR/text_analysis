# analyze4.py „UMAP Embedding Space: Errors vs Correct"

import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
import os
import base

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

MODELS = [
    "class_fcnn_bge"
]

print(f"Records: {len(df):,}")


s_sentence_transformer = 'BAAI/bge-base-en-v1.5'
s_senctence_tranformer_short = '_bge'


# Embeddings laden (aus analyze2.py)
emb_file = os.path.join(base.figures_dir, "11_all_text_embeddings" + s_senctence_tranformer_short + ".npy")
if os.path.exists(emb_file):
    emb = np.load(emb_file)
else:
    print("Embeddings nicht gefunden – werden neu berechnet...")
    from sentence_transformers import SentenceTransformer
    model_st = SentenceTransformer(s_sentence_transformer)
    emb = model_st.encode(df["text_"].tolist(), show_progress_bar=True)
    np.save(emb_file, emb)

print(f"Embeddings shape: {emb.shape}")


# ============================================================
# UMAP
# ============================================================
print("\n" + "=" * 80 + "\n1. UMAP Projektion\n" + "=" * 80)

umap_cache = os.path.join(base.figures_dir, "13_umap_cache" + s_senctence_tranformer_short + ".npy")

if os.path.exists(umap_cache):
    print("UMAP Cache gefunden – wird geladen...")
    emb_2d_umap = np.load(umap_cache)
else:
    print("UMAP wird berechnet (kann einige Minuten dauern)...")
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )
    emb_2d_umap = umap_model.fit_transform(emb)
    np.save(umap_cache, emb_2d_umap)
    print("UMAP gespeichert.")

df["umap_x"] = emb_2d_umap[:, 0]
df["umap_y"] = emb_2d_umap[:, 1]
# nur wenn alle Modelle richtig liegen, wird der Punkt grün
df["any_error"] = (df[MODELS].sum(axis=1) < len(MODELS)).astype(int)


# ============================================================
# PLOT
# ============================================================
print("\n" + "=" * 80 + "\n2. Plot\n" + "=" * 80)

colors = df["any_error"].map({0: "#2ecc71", 1: "#e74c3c"})
labels = df["any_error"].map({0: "All correct", 1: "At least 1 error"})


def wrap_text(text, n=10):
    words = str(text).split()
    lines = [" ".join(words[i:i+n]) for i in range(0, len(words), n)]
    return "<br>".join(lines)

#hovertext = df["text_"].apply(wrap_text)

hovertext = df.apply(
    lambda row: f"<b>Label: {row['label']}</b><br><br>" + wrap_text(row["text_"]),
    axis=1
)



fig = go.Figure(
    data=go.Scatter(
        x=df["umap_x"],
        y=df["umap_y"],
        mode="markers",
        marker=dict(size=3, color=colors),
        hovertext=hovertext, #df["text_"],
        hoverinfo="text",
        text=labels,
    )
)

fig.update_layout(
    height=800,
    title_text="UMAP Embedding Space: Errors vs Correct",
    xaxis_title="UMAP 1",
    yaxis_title="UMAP 2",
)

fig.write_html(os.path.join(base.figures_dir, "13_umap_embedding_space.html"))
fig.show()
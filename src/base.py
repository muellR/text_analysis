import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


# Ordner-Pfade
base_dir = os.path.dirname(os.path.abspath(__file__))
resources_dir = os.path.abspath(os.path.join(base_dir, '..', 'resources'))
predictions_dir = os.path.abspath(os.path.join(base_dir, '..', 'generated', 'predictions'))
figures_dir = os.path.abspath(os.path.join(base_dir, '..', 'generated', 'figures'))
models_dir = os.path.abspath(os.path.join(base_dir, '..', 'generated', 'models'))


# ─────────────────────────────────────────────
# Funktion um Ordner-Pfade neu zu erstellen
# ─────────────────────────────────────────────
def reinitialize_folders(folders, drop_existing=False):
    # prüfen, ob folders eine Liste ist, wenn nicht, in eine Liste umwandeln
    # sonst macht es für jeden Buchstaben im String einen Ordner, was nicht gewollt ist
    if not isinstance(folders, list):
        folders = [folders]

    for p in folders:
        if Path(p).exists() and drop_existing:
            shutil.rmtree(p)
        Path(p).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
# Daten laden und splitten in Train und Test mit Embeddings
# ─────────────────────────────────────────────────────────
def load_and_split_data_old(s_file_name="fake reviews dataset.csv"):
    df = pd.read_csv(os.path.join(resources_dir, s_file_name))

    # one-hot encoding für category
    df = pd.get_dummies(df, columns=['category'], dtype=float)

    # die Spalte "label" in 0=Fake und 1=Real umwandeln
    df['label'] = df['label'].map({'OR': 1, 'CG': 0})

    # die Spalte rating normalisieren (0-1)
    df['rating_norm'] = (df['rating'] - 1) / 4

    # wenn es das Feld sentiment gibt, dann normalisieren (0-1)
    if 'sentiment' in df.columns:
        df['sentiment'] = (df['sentiment'] - 1) / 4

    # text_ in Vektoren umwandeln und an df anhängen
    df = add_embeddings(df)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"] # stratify nach label, damit die Verteilung in Train und Test gleich ist
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ─────────────────────────────────────────────
# Daten laden und splitten in Train und Test
# ─────────────────────────────────────────────
def load_and_split_data(s_file_name="fake reviews dataset.csv"):
    df = pd.read_csv(os.path.join(resources_dir, s_file_name))

    # IDs erzeugen für spätere Identifikation der falsch deklarierten
    df["id"] = range(1, len(df) + 1)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"] # stratify nach label, damit die Verteilung in Train und Test gleich ist
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)




# ─────────────────────────────────────────────
# Embedings von Description rechnen und bei df anfügen
# ─────────────────────────────────────────────
def add_embeddings(df):
    reinitialize_folders([models_dir], drop_existing=False)

    # wenn File description_embeddings.npy nicht vorhanden
    if not os.path.exists(os.path.join(models_dir, "embeddings.npy")):
        from sentence_transformers import SentenceTransformer
        from concurrent.futures import ThreadPoolExecutor
        import threading

        embedder = SentenceTransformer('all-MiniLM-L12-v2')
        def encode_batch(args):
            idx, batch = args
            worker = threading.current_thread().name
            print(f"[{worker}] Batch {idx+1}/{len(batches)} ({len(batch)} records)")
            return embedder.encode(batch)

        texts = df['text_'].tolist()
        batch_size = 512
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(encode_batch, enumerate(batches)))

        embeddings = np.vstack(results)

        # save embeddings as npy file
        np.save(os.path.join(models_dir, "embeddings.npy"), embeddings)
    else:
        # laod file
        embeddings = np.load(os.path.join(models_dir, "embeddings.npy"))


    # die embeddings an den DataFrame anhängen und zurückgeben
    embedding_cols = [f"emb_{i}" for i in range(384)]
    df = pd.concat([df, pd.DataFrame(embeddings, index=df.index, columns=embedding_cols)], axis=1)
    print(df.shape)
    return df




# 🔧 Define a helper function to plot learning curves
def plot_learning_curves(history):
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    ax = axs[0]
    ax.plot(history.history['loss'], label="training")
    if history.history.get('val_loss') is not None:
        ax.plot(history.history['val_loss'], label="validation")
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend()
    ax.grid()

    ax = axs[1]
    ax.plot(history.history['accuracy'], label="training")
    if history.history.get('val_accuracy') is not None:
        ax.plot(history.history['val_accuracy'], label="validation")
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend()
    ax.grid()

    return fig
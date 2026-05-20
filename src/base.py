import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# Ordner-Pfade
base_dir = os.path.dirname(os.path.abspath(__file__))
resources_dir = os.path.abspath(os.path.join(base_dir, '..', 'resources'))
tmp_dir = os.path.abspath(os.path.join(base_dir, '..', 'generated', 'tmp'))
figures_dir = os.path.abspath(os.path.join(base_dir, '..', 'generated', 'figures'))
models_dir = os.path.abspath(os.path.join(base_dir, '..', 'generated', 'models'))


# ─────────────────────────────────────────────
# Funktion um Ordner-Pfade neu zu erstellen
# ─────────────────────────────────────────────
def reinitialize_folders(folders, drop_existing=False):
    for p in folders:
        if Path(p).exists() and drop_existing:
            shutil.rmtree(p)
        Path(p).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Daten laden und splitten in Train und Test
# ─────────────────────────────────────────────
def load_and_split_data(b_add_embeddings=False):
    df = pd.read_csv(os.path.join(resources_dir, "fake reviews dataset.csv"))

    if b_add_embeddings:
        df = add_embeddings(df)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
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
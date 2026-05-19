import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# Ordner-Pfade
base_dir = os.path.dirname(os.path.abspath(__file__))
resources_dir = os.path.abspath(os.path.join(base_dir, '..', 'resources'))
tmp_dir = os.path.abspath(os.path.join(base_dir, '..', 'generated', 'tmp'))
figures_dir = os.path.abspath(os.path.join(base_dir, '..', 'generated', 'figures'))
models_dir = os.path.abspath(os.path.join(base_dir, '..', 'generated', 'models'))


# Funktion um Ordner-Pfade neu zu erstellen
def reinitialize_folders(folders, drop_existing=False):
    for p in folders:
        if Path(p).exists() and drop_existing:
            shutil.rmtree(p)
        Path(p).mkdir(parents=True, exist_ok=True)



def load_and_split_data():
    df = pd.read_csv(os.path.join(resources_dir, "fake reviews dataset.csv"))

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
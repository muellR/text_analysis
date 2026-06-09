# Dieses Skript liest alle correct_predictions_*.csv und wrong_predictions_*.csv Dateien im predictions-Verzeichnis ein, 
# #kombiniert sie und speichert das Ergebnis in combined_predictions.csv.


import pandas as pd
import os
import re

from torch import cat
import base
from pathlib import Path

base_dir = Path(base.predictions_dir)

# combined_predictions.csv auschliessen, falls vorhanden
files = [f for f in base_dir.glob("*.csv") if not f.name == "combined_predictions.csv"]


dfs = [] # Sammelliste für alle DataFrames
name_cols = set()

for f in files:
    df = pd.read_csv(f)

    # Klassenname extrahieren: den Teil zwischen "correct_predictions_" oder "wrong_predictions_" und ".csv" extrahieren
    s_base = os.path.basename(f)
    name = re.sub(r"^(correct|wrong)_predictions_|\.csv$", "", s_base)

    # 1 = correct, 0 = wrong
    correct = 1 if s_base.startswith("correct_predictions") else 0

    df = df[["id", "category", "rating", "label", "text_"]].copy()
    df["name"] = name
    df["value"] = correct

    dfs.append(df)

    # name-Spalte einmalig merken
    name_cols.add(name)

# Alle DataFrames untereinander zusammenführen (id's doppelt)
all_df = pd.concat(dfs, ignore_index=True)

# Basisdaten pro id (angenommen identisch in allen Dateien)
basis = all_df[["id", "category", "rating", "label", "text_"]].drop_duplicates()

# Pivot: pro name eine Spalte (0/1)
wide = all_df.pivot_table(
    index="id",         # eine Zeile pro id
    columns="name",     # jeder Klassenname wird eine eigene Spalte
    values="value",
    aggfunc="max"
).reset_index()

# Basisdaten mit den pivotierten Daten zusammenführen
result = basis.merge(wide, on="id", how="left")

# NaN-Werte (id kam in einer Datei nicht vor) → 0 setzen und als Integer speichern
result[list(name_cols)] = result[list(name_cols)].fillna(0).astype(int)

# füge weitere Spalten hinzu, z.B. die Anzahl der correct_predictions pro id
result["correct_count"] = result[list(name_cols)].sum(axis=1)

# Ergebnis speichern
result.to_csv(os.path.join(base.predictions_dir, "combined_predictions.csv"), index=False)
print("Combined predictions saved to:", os.path.join(base.predictions_dir, "combined_predictions.csv"))
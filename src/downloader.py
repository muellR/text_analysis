import os
import shutil
from pathlib import Path
import kagglehub
import base

# Parent-Verzeichnis in die Liste der Verzeichnisse, die Python durchsucht (für Run in vsc)
#sys.path.insert(0, str(Path(__file__).resolve().parent))

# Download der neusten Version
path = Path(kagglehub.dataset_download("muqaddasejaz/fake-reviews-dataset", force_download=True))

# Zielverzeichnis sicherstellen
os.makedirs(base.resources_dir, exist_ok=True)

for f in path.iterdir():
    if f.is_file():
        shutil.copy(str(f), os.path.join(base.resources_dir, f.name))
        print("Path to dataset file:", os.path.join(base.resources_dir, f.name))
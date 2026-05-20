from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# GPU nutzen falls verfügbar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Inference-Modus

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def collate_fn(batch):
    return tokenizer(
        batch,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True  # Texte im Batch auf gleiche Länge padden
    )

def analyse_sentiment_batch(df: pd.DataFrame, text_col: str, batch_size: int = 32) -> pd.DataFrame:
    dataset = TextDataset(df[text_col].fillna("").tolist())
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_sentiment = []
    all_konfidenz = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Sentiment-Analyse"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

            sentiment = torch.argmax(probs, dim=-1) + 1  # Shape: (batch_size,)
            konfidenz = probs[torch.arange(len(sentiment)), sentiment - 1]

            all_sentiment.extend(sentiment.cpu().tolist())
            all_konfidenz.extend(konfidenz.cpu().tolist())

    df = df.copy()
    df["sentiment"] = all_sentiment
    df["konfidenz"] = all_konfidenz

    return df

# Verwendung
df = pd.read_csv("MAS/text_analysis/resources/fake reviews dataset.csv")#.head(100)  # Beispiel: nur die ersten 100 Zeilen analysieren
df_result = analyse_sentiment_batch(df, text_col="text_", batch_size=32)
df_result.to_csv("MAS/text_analysis/resources/fake reviews dataset_senti.csv", index=False)

print("done")
# hier sollen die Inputdaten in einem Modell gelern werden:
# 1. Embedden des Text
# 2. lernen eines Neuronalen Nezt (FCCN)
# 3. Vorhersage und Zusammenfassung (Konfusionsmatrix)
#   Option: Zusätliches Lernen mit abgeleitetem Sentiment und Vergleich der beiden outcomes

import base
import pandas as pd
import os
import numpy as np

train_df, test_df = pd, pd
train_df, test_df = base.load_and_split_data(b_add_embeddings=True)

print(train_df.head())
print(test_df.head())
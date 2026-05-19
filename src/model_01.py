import base

import pandas as pd


train_df, test_df = pd, pd

train_df, test_df = base.load_and_split_data()

print(train_df.head())
print(test_df.head())
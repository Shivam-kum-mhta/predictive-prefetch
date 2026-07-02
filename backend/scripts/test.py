import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('./dataset/MINDsmall_train_behaviours.tsv')

print(df)
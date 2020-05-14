import pandas as pd

train_df = pd.read_csv('mini-train.csv', sep=',')
train_df.to_csv('train.tsv', index=False, sep='\t')

valid_df = pd.read_csv('mini-valid.csv', sep=',')
valid_df.to_csv('valid.tsv', index=False, sep='\t')

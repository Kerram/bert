import pandas as pd

test_df = pd.read_csv('test.csv', sep=',').head(1)
test_df.to_csv('test.tsv', index=False, sep='\t')

train_df = pd.read_csv('train.csv', sep=',')
train_df.to_csv('train.tsv', index=False, sep='\t')

valid_df = pd.read_csv('valid.csv', sep=',')
valid_df.to_csv('valid.tsv', index=False, sep='\t')

import pandas as pd


def preprocess(df):
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.replace(['\(', '\)'], [' ', ' '], regex=True)
    df = df.replace(['<NULL>'], [''], regex=True)

    print("Preprocessing done.")

    return df


test_df = pd.read_csv('test.csv', sep=',').head(8)
test_df = preprocess(test_df)
test_df.to_csv('preprocessed_test.tsv', index=False, sep='\t')

train_df = pd.read_csv('train.csv', sep=',')
train_df = preprocess(train_df)
train_df.to_csv('preprocessed_train.tsv', index=False, sep='\t')

valid_df = pd.read_csv('valid.csv', sep=',')
valid_df = preprocess(valid_df)
valid_df.to_csv('preprocessed_valid.tsv', index=False, sep='\t')

valid_df = valid_df.head(50_000)
valid_df.to_csv('preprocessed_valid_mini.tsv', index=False, sep='\t')

import pandas as pd


train_df = pd.read_csv('train.csv', sep=',')
train_df = train_df.replace(['\(', '\)'], ['', ''], regex=True)

counter = {}

for index, row in train_df.iterrows():
    pair = (row.goal, row.tactic_id)
    if pair not in counter:
        counter[pair] = 1
    else:
        counter[pair] += 1

    if index % 10000 == 0:
        print("DONE %d" % (index,))

weights = []
for index, row in train_df.iterrows():
    pair = (row.goal, row.tactic_id)
    weights.append(1/(counter[pair] // 8))

    if index % 10000 == 0:
        print("DONE %d" % (index,))

train_df = train_df.assign(weight=pd.Series(weights).values)
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df.to_csv('new_train.tsv', index=False, sep='\t')

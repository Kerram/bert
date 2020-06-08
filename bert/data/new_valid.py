import pandas as pd

# Enhance function by the courtesy of Filip Mikina.
def enhance(df):
    last_row_ind = 0
    seen_false = False
    seen_true = False
    for ind, row in df.iterrows():
        if ind % 10000 == 0:
            print("DONE", ind)

        if not row['is_negative']:
            if seen_true and seen_false:
                seen_false = False
                seen_true = False

                if (ind - last_row_ind) % 8 != 0:
                    print("ERROR", ind)

                df.loc[last_row_ind:(ind - 1), 'weight'] = 1 / ((ind - last_row_ind) // 8)
                last_row_ind = ind

            seen_false = True
        else:
            seen_true = True

    # Update last
    ind = len(df)

    if (ind - last_row_ind) % 8 != 0:
        print("ERROR", ind)

    df.loc[last_row_ind:(ind - 1), 'weight'] = 1 / ((ind - last_row_ind) // 8)
    return df


df = pd.read_csv('valid.csv', sep=',')
df = enhance(df)
df = df.replace(['\(', '\)'], ['', ''], regex=True)

df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('new_valid.tsv', index=False, sep='\t')

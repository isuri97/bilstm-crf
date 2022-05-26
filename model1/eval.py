import pandas as pd
import matplotlib.pyplot as plt
from bi_lstm_crf.app import WordsTagger

# the training losses are saved in the model_dir
def main():
    # df = pd.read_csv("loss.csv")
    # df[["train_loss", "val_loss"]].ffill().plot(grid=True)
    # plt.show()

    model = WordsTagger(model_dir="../model1")

    tags, sequences = model([["Australia", "is", "a","country"]])  # CHAR-based model
    print(tags)
    print(sequences)


def process(df):
    i = 0
    col = []
    for index, row in df.iterrows():
        if '#' in row['TOKEN']:
            col.append(None)
            if '#' not in df.iloc[index + 1]['TOKEN']:
                i += 1
            continue
        else:
            col.append(i)

    df["sent_id"] = col
    df = df[df['sent_id'].notna()]
    df.sent_id = df['sent_id'].astype(int)
    df2 = df[['TOKEN', 'NE-COARSE-LIT', 'sent_id']]
    df2.to_csv('new.txt')

if __name__ == "__main__":
    main()
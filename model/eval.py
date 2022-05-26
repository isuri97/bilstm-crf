import pandas as pd
import matplotlib.pyplot as plt
from bi_lstm_crf.app import WordsTagger

# the training losses are saved in the model_dir
def main():
    # df = pd.read_csv("loss.csv")
    # df[["train_loss", "val_loss"]].ffill().plot(grid=True)
    # plt.show()

    model = WordsTagger(model_dir="../model")
    tags, sequences = model(["市领导到成都高新区进行考察"])  # CHAR-based model
    print(tags)
    print(sequences)

if __name__ == "__main__":
    main()
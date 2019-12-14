import pandas as pd;

if __name__ == "__main__":
    data = pd.read_csv("data/audio/X_sig_train.txt", sep=",", header=None)
    print(data.shape)
    data = pd.read_csv("data/text/X_sig_train.txt", sep=",", header=None)
    print(data.shape)
    data = pd.read_csv("data/video/X_sig_train.txt", sep=",", header=None)
    print(data.shape)

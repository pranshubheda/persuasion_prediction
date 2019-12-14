import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def late_fusion_unimodal_embeddings():
    """Read each individual modality embedding of the train data
    Returns:
        X, Y the concatenated embedding vectors and labels
    """
    data_audio = pd.read_csv("data/intermediate_output/train/intermediate_output_audio.csv", sep=",", header=None)
    X_audio = data_audio.values

    data_video = pd.read_csv("data/intermediate_output/train/intermediate_output_video.csv", sep=",", header=None)
    X_video = data_video.values

    data_text = pd.read_csv("data/intermediate_output/train/intermediate_output_text.csv", sep=",", header=None)
    X_text = data_text.values

    labels = pd.read_csv("data/audio/y_train.txt", sep=",", header=None)
    Y = labels.values

    X = np.concatenate((X_audio, X_text, X_video), axis=1)
    return X, Y

def train_deep_fusion_model(X, Y):
    """Train intermediate output model on the training
    data embeddings.
    """
    input_dimensions = X.shape[1]
    model = Sequential()
    model.add(Dense(10, input_dim=input_dimensions, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X,Y,epochs=1,verbose=1)
    model.summary()
    # serialize model to JSON
    model_json = model.to_json()
    with open("models/intermediate_output_deep_fusion_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/intermediate_output_deep_fusion_model.h5")
    print("Saved Intermediate output model to disk") 

if __name__ == "__main__":
    #concatenate the intermediate output features
    X, Y = late_fusion_unimodal_embeddings()
    #train the model
    train_deep_fusion_model(X, Y)
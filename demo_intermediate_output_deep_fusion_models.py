from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os
import pandas as pd
from keras.models import Model

def late_fusion_unimodal_embeddings():
    data_audio = pd.read_csv("demo_intermediate_output/intermediate_output_audio.csv", sep=",", header=None)
    X_audio = data_audio.values

    data_video = pd.read_csv("demo_intermediate_output/intermediate_output_video.csv", sep=",", header=None)
    X_video = data_video.values

    data_text = pd.read_csv("demo_intermediate_output/intermediate_output_text.csv", sep=",", header=None)
    X_text = data_text.values

    labels = pd.read_csv("data/audio/y_test.txt", sep=",", header=None)
    Y = labels.values

    X = np.concatenate((X_audio, X_text, X_video), axis=1)
    return X, Y

def test_intermediate_output_deep_fusion(X, Y):
    # load json and create model
    json_file = open('trained_models/models/intermediate_output_deep_fusion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("trained_models/models/intermediate_output_deep_fusion_model.h5")
    print("Loaded Intermediate output model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y, verbose=1)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    predicted_Y = loaded_model.predict(X, verbose=1)
    print(np.round(predicted_Y, 3))

if __name__ == "__main__":
    #concatenate the intermediate output features
    X, Y = late_fusion_unimodal_embeddings()
    #test the intermediate output model
    test_intermediate_output_deep_fusion(X, Y)
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os
import pandas as pd
from keras.models import Model

def test_audio_model():
    #read the test data for a model
    data = pd.read_csv("data/audio/X_sig_test.txt", sep=",", header=None)
    labels = pd.read_csv("data/audio/y_test.txt", sep=",", header=None)
    X = data.values
    Y = labels.values

    # load json and create model
    json_file = open('models/audio_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/audio_model.h5")
    print("Loaded audio model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    #predict and use the intermediate results
    intermediate_layer_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[1].output)
    intermediate_output = intermediate_layer_model.predict(X)
    np.savetxt('data/intermediate_output/test/intermediate_output_audio.csv', intermediate_output, delimiter=',')

def test_video_model():
    #read the test data for a model
    data = pd.read_csv("data/video/X_sig_test.txt", sep=",", header=None)
    labels = pd.read_csv("data/video/y_test.txt", sep=",", header=None)
    X = data.values
    Y = labels.values
    
    # load json and create model
    json_file = open('models/video_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/video_model.h5")
    print("Loaded video model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    #predict and use the intermediate results
    intermediate_layer_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[1].output)
    intermediate_output = intermediate_layer_model.predict(X)
    np.savetxt('data/intermediate_output/test/intermediate_output_video.csv', intermediate_output, delimiter=',')

def test_text_model():
    #read the test data for a model
    data = pd.read_csv("data/text/X_sig_test.txt", sep=",", header=None)
    labels = pd.read_csv("data/text/y_test.txt", sep=",", header=None)
    X = data.values
    Y = labels.values
    
    # load json and create model
    json_file = open('models/text_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/text_model.h5")
    print("Loaded text model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    #predict and use the intermediate results
    intermediate_layer_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[1].output)
    intermediate_output = intermediate_layer_model.predict(X)
    np.savetxt('data/intermediate_output/test/intermediate_output_text.csv', intermediate_output, delimiter=',')

if __name__ == "__main__":
    test_audio_model()
    test_video_model()
    test_text_model()
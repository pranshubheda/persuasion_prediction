from keras.models import Sequential
from keras.layers import Dense, Activation
import keras_metrics as metrics
import pandas as pd
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.models import Model
import numpy as np

"""Final project - Persuasion prediction
Pranshu Bheda & Akshay Kshirsagar
"""

def build_classifier(data, labels):
    """Create a unimodal classifier using
    the data and the corresponding labels from
    the dataset.
    """
    X = data.values
    Y = labels.values
    intermediate_output, model = build_unimodal_classifier(X, Y)
    return intermediate_output, model

def build_audio_classifier():
    """Build a classifier for predicting 
    persuasion using the audio modality of the 
    data set.
    """    
    data = pd.read_csv("data/audio/X_sig_train.txt", sep=",", header=None)
    labels = pd.read_csv("data/audio/y_train.txt", sep=",", header=None)
    intermediate_output, model = build_classifier(data, labels)    
    np.savetxt('data/intermediate_output/train/intermediate_output_audio.csv', intermediate_output, delimiter=',')
    # serialize model to JSON
    model_json = model.to_json()
    with open("models/audio_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/audio_model.h5")
    print("Saved audio model to disk")
    
def build_video_classifier():
    """Build a classifier for predicting 
    persuasion using the video modality of the 
    data set.
    """
    data = pd.read_csv("data/video/X_sig_train.txt", sep=",", header=None)
    labels = pd.read_csv("data/video/y_train.txt", sep=",", header=None)
    intermediate_output, model = build_classifier(data, labels)    
    np.savetxt('data/intermediate_output/train/intermediate_output_video.csv', intermediate_output, delimiter=',')
    # serialize model to JSON
    model_json = model.to_json()
    with open("models/video_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/video_model.h5")
    print("Saved video model to disk")

def build_text_classifier():
    """Build a classifier for predicting 
    persuasion using the text modality of the 
    data set.
    """
    data = pd.read_csv("data/text/X_sig_train.txt", sep=",", header=None)
    labels = pd.read_csv("data/text/y_train.txt", sep=",", header=None)
    intermediate_output, model = build_classifier(data, labels)    
    np.savetxt('data/intermediate_output/train/intermediate_output_text.csv', intermediate_output, delimiter=',')
    # serialize model to JSON
    model_json = model.to_json()
    with open("models/text_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/text_model.h5")
    print("Saved text model to disk")

def build_unimodal_classifier(X, Y):
    """We traina neural network for each modality.
    This function trains a model using the training
    data vectors 'X' and the corresponding labels 'Y'.
    We reuse this function for different modalities.
    """
    input_dimensions = X.shape[1]
    model = unimodal_model(input_dimensions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X,Y,epochs=1,verbose=1)
    y_pred = model.predict_classes(X)

    #obtain the embedding of the intermediate layer
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[1].output)
    intermediate_output = intermediate_layer_model.predict(X)
    model.summary()
    return intermediate_output, model

def unimodal_model(input_dimensions):
    """Build a Sequential 2 layer neural network
    2 Dense fully connected nodes with 10 nodes each.    
    Returns:
        model -- The model object
    """
    model = Sequential()
    model.add(Dense(10, input_dim=input_dimensions, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    return model

if __name__ == "__main__":
    print("************TRAIN AUDIO CLASSIFIER****************")
    build_audio_classifier()
    print("************TRAIN VIDEO CLASSIFIER****************")
    build_video_classifier()
    print("************TRAIN TEXT CLASSIFIER****************")
    build_text_classifier()
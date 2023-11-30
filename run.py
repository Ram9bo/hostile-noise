import io
import os
import zipfile

import librosa
import librosa.display
import numpy as np
import requests
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from tqdm import tqdm
import json


# Function to load and preprocess a single audio file
def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = np.concatenate([spectrogram, spectrogram_db])
    # Reshape the 2D spectrogram to have a single channel
    reshaped_spectrogram = np.expand_dims(spectrogram, axis=-1)
    return reshaped_spectrogram


# Function to extract class label from file name
def extract_class_label(file_name):
    # Label is the last part
    return file_name.split(".")[0].split('-')[3]


def download_data():
    # Set the URL for the zip file
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"

    extract_folder = "data"
    if not os.listdir(extract_folder):
        # Download the zip file
        print("starting download.")
        response = requests.get(url)
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        print("starting extracting.")
        # Extract the contents to the specified folder
        zip_file.extractall(extract_folder)

        # Close the zip file
        zip_file.close()
        print("Download and extraction complete.")
    else:
        print("Data folder is not empty. Skipping download.")


def get_data():
    # Specify the path to the folder containing WAV files
    data_folder = "data/ESC-50-master/audio"

    # List to store spectrograms and corresponding labels
    spectrograms = []
    labels = []

    hostility = {}
    with open('hostility.json', 'r') as json_file:
        hostility = json.load(json_file)

    # Iterate through each WAV file in the folder
    for file_name in tqdm(os.listdir(data_folder)):
        if file_name.endswith(".wav"):
            # Load and preprocess audio
            audio_path = os.path.join(data_folder, file_name)
            spectrogram = preprocess_audio(audio_path)

            # Extract label
            label = extract_class_label(file_name)
            hostile = hostility[label]["hostile"]

            # Append to the lists
            spectrograms.append(spectrogram)
            labels.append(hostile)

    # Convert lists to numpy arrays
    spectrograms = np.array(spectrograms)
    labels = np.array(labels)

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms, labels)).batch(16)
    return dataset


def train(data, model=None):
    if model is None:
        input_shape = data.element_spec[0].shape
        input_shape = [i for i in input_shape if i is not None]
        model = construct_model(input_shape=input_shape)

    hist = model.fit(data, epochs=5)

    return model, hist


def construct_model(input_shape):
    model = Sequential()

    # Apply convolutional layers
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the 2D data
    model.add(Flatten())

    model.add(Dense(128))
    model.add(Dropout(0.1))
    model.add(Dense(128))
    model.add(Dropout(0.1))
    model.add(Dense(64))
    model.add(Dropout(0.1))
    model.add(Dense(32))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # TODO: try other architectures maybe hyperparameter-optimization

    return model


if __name__ == "__main__":
    download_data()
    data = get_data()
    train_data, test_data = tf.keras.utils.split_dataset(data, left_size=0.9)

    model, hist = train(train_data)

    score = model.evaluate(test_data, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])




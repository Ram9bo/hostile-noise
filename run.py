import requests
import zipfile
import io
import os
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm


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
def extract_label(file_name):
    # Label is the last part
    label_str = file_name.split(".")[0].split('-')[3]
    return int(label_str)


def download_data():
    # Set the URL for the zip file
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"

    extract_folder = "data"
    if not os.listdir(extract_folder):
        # Download the zip file
        response = requests.get(url)
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

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

    # Iterate through each WAV file in the folder
    for file_name in tqdm(os.listdir(data_folder)):
        if file_name.endswith(".wav"):
            # Load and preprocess audio
            audio_path = os.path.join(data_folder, file_name)
            spectrogram = preprocess_audio(audio_path)

            # Extract label
            label = extract_label(file_name)

            # Append to the lists
            spectrograms.append(spectrogram)
            labels.append(label)

    # Convert lists to numpy arrays
    spectrograms = np.array(spectrograms)
    labels = np.array(labels)

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms, labels)).batch(16)
    return dataset


def train(data, model=None):
    if model is None:
        # TODO: dynamically determine input and output shapes
        model = construct_model()

    # TODO: add validation
    hist = model.fit(data, epochs=20)

    return hist


def construct_model(input_shape=(128 * 2, 216, 1), num_classes=50):
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

    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # TODO: try other architectures

    return model


if __name__ == "__main__":
    download_data()
    train(get_data())

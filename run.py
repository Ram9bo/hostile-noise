import io
import json
import os
import random
import zipfile

import librosa
import librosa.display
import numpy as np
import requests
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras_tuner.tuners import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tqdm import tqdm
import json

input_shape = None


# Function to load and preprocess a single audio file
def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = np.concatenate([spectrogram, spectrogram_db])
    # Reshape the 2D spectrogram to have a single channel
    reshaped_spectrogram = np.expand_dims(spectrogram, axis=-1)
    return reshaped_spectrogram / 255


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

    # Lists to store spectrograms and labels for each class
    true_class_spectrograms = []
    false_class_spectrograms = []

    # Iterate through each WAV file in the folder
    files = os.listdir(data_folder)
    random.shuffle(files)
    for file_name in tqdm(files):
        if file_name.endswith(".wav"):
            # Load and preprocess audio
            audio_path = os.path.join(data_folder, file_name)
            spectrogram = preprocess_audio(audio_path)

            # Extract label
            label = extract_class_label(file_name)
            hostile = int(hostility[label]["hostile"])

            # Append to the lists based on class
            if hostile:
                true_class_spectrograms.append(spectrogram)
            else:
                false_class_spectrograms.append(spectrogram)

    # Undersample the majority class (false class) to balance the dataset
    min_class_size = min(len(true_class_spectrograms), len(false_class_spectrograms))
    true_class_spectrograms = random.sample(true_class_spectrograms, min_class_size)
    false_class_spectrograms = random.sample(false_class_spectrograms, min_class_size)

    # Combine the balanced data
    balanced_spectrograms = true_class_spectrograms + false_class_spectrograms
    labels = [1] * min_class_size + [0] * min_class_size

    # Shuffle the data
    combined_data = list(zip(balanced_spectrograms, labels))
    random.shuffle(combined_data)
    balanced_spectrograms[:], labels[:] = zip(*combined_data)

    # Convert lists to numpy arrays
    spectrograms = np.array(balanced_spectrograms)
    labels = np.array(labels)

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms, labels)).batch(16)

    i_shape = dataset.element_spec[0].shape
    i_shape = [i for i in i_shape if i is not None]

    global input_shape
    input_shape = i_shape
    return dataset


def construct_model(num_conv_layers, conv_filters, num_dense_layers, dense_neurons, dropout, learning_rate):
    global input_shape
    assert input_shape is not None

    model = Sequential()

    model.add(Conv2D(conv_filters, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    for _ in range(num_conv_layers):
        model.add(Conv2D(conv_filters, kernel_size=(3, 3), activation='relu'))

    model.add(Flatten())

    for _ in range(num_dense_layers):
        model.add(Dense(dense_neurons, activation='relu'))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model


def build_model_tuned(hp):
    global input_shape
    assert input_shape is not None

    model = Sequential()

    num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=5)
    conv_filters = hp.Int('conv_filters', min_value=8, max_value=64, step=8)

    model.add(Conv2D(conv_filters, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    for _ in range(num_conv_layers):
        model.add(Conv2D(conv_filters, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=5)
    dense_neurons = hp.Int('dense_neurons', min_value=16, max_value=128, step=16)
    dropout = hp.Float('dropout', min_value=0.0, max_value=0.3, step=0.1)

    for _ in range(num_dense_layers):
        model.add(Dense(dense_neurons, activation='relu'))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        # TODO: write custom F1 score and include it, then we can also tune for it
    )

    return model


def tune(train_data, val_data):
    tuner = BayesianOptimization(
        build_model_tuned,
        objective='val_accuracy',
        directory='tuner_logs',
        project_name='audio_classification',
        executions_per_trial=3,
        max_trials=35,
        overwrite=False
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    tuner.search(train_data,
                 epochs=10,
                 validation_data=val_data,
                 callbacks=[early_stopping])

    best_hps = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values

    return best_hps


if __name__ == "__main__":
    tune_more = True  # Whether to do more tuning or use the best known hyperparameters immediately
    download_data()
    data = get_data()

    total_size = tf.data.experimental.cardinality(data).numpy()
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    # Split the dataset
    train_data = data.take(train_size)
    test_data = data.skip(train_size)
    val_data = test_data.take(val_size)
    test_data = test_data.skip(val_size)

    if tune_more:
        best_hyperparameters = tune(train_data, val_data)
        with open("best_params.json", "w") as json_file:
            json.dump(best_hyperparameters, json_file)
    else:
        with open("best_params.json", "r") as json_file:
            best_hyperparameters = json.load(json_file)

    # Print the best hyperparameters
    print("Best Hyperparameters:")
    print(best_hyperparameters)

    tf.keras.backend.clear_session()

    with open("eval.json", "r") as json_file:
        metrics = json.load(json_file)

    final_model = construct_model(**best_hyperparameters)
    final_model.fit(train_data, epochs=10, validation_data=val_data, verbose=1)

    final_model.save('Model/model.keras')

    score = final_model.evaluate(test_data, verbose=0)
    metrics.append(score)

    with open("eval.json", "w") as json_file:
        json.dump(metrics, json_file)

    score = np.mean(np.array(metrics), axis=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test precision:', score[2])
    print('Test recall:', score[3])
    print('Test F1:', 2 * (score[2] * score[3]) / (score[2] + score[3]))

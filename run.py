import requests
import zipfile
import io
import os
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential


def download_data():
    # Set the URL for the zip file
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"

    # Set the directory and folder names
    download_directory = "your_download_directory"
    extract_folder = "data"

    # Create the download directory if it doesn't exist
    os.makedirs(download_directory, exist_ok=True)

    # Check if the "data" folder is empty
    data_folder = os.path.join(download_directory, extract_folder)
    if not os.listdir(data_folder):
        # Download the zip file
        response = requests.get(url)
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        # Extract the contents to the specified folder
        zip_file.extractall(data_folder)

        # Close the zip file
        zip_file.close()
        print("Download and extraction complete.")
    else:
        print("Data folder is not empty. Skipping download.")


def get_data():
    # TODO: read wav files and convert them to spectrograms (or something else)
    pass


def train(data, model=None):
    if model is None:
        # TODO: dynamically determine input and output shapes
        model = construct_model()

    # TODO: add validation
    hist = model.fit(data)

    return hist


def construct_model(input_shape=(128, 128), num_classes=50):
    model = Sequential()

    # Apply convolutional layers
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the 2D data
    model.add(Flatten())

    model.add(Dense(128))
    model.add(Dense(num_classes))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # TODO: try other architectures

    return model


if __name__ == "__main__":
    download_data()

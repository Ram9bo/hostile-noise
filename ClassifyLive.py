import io
import os


from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from tqdm import tqdm

import sounddevice as sd
from scipy.io.wavfile import write

def process_audio(y, sr):
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = np.concatenate([spectrogram, spectrogram_db])
    # Reshape the 2D spectrogram to have a single channel
    reshaped_spectrogram = np.expand_dims(spectrogram, axis=-1)

    return reshaped_spectrogram


if __name__ == "__main__":

    model = keras.models.load_model('Model/model.keras')
    sr = 22050  # Sample rate
    seconds = 5  # Duration of recording

    while(True):
    
        y = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
        sd.wait()
        write('output.wav', sr, y)
        y, sr = librosa.load('output.wav')
        spectrogram = process_audio(y, sr)
        spectrograms = np.array(spectrogram)
        spectrograms = spectrograms[None,:,:,:]
        test = tf.convert_to_tensor(spectrograms, dtype=tf.float32)
        score = model.predict(test, verbose=0)
        print(np.argmax(score))
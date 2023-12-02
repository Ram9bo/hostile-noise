import time

import keras
import keyboard
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import tensorflow as tf
from scipy.io.wavfile import write


def process_audio(y, sr):
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = np.concatenate([spectrogram, spectrogram_db])
    # Reshape the 2D spectrogram to have a single channel
    reshaped_spectrogram = np.expand_dims(spectrogram, axis=-1)

    return reshaped_spectrogram


def get_sound():
    ext = 1  # exit flag
    sr = 22050  # Sample rate
    seconds = 5  # Duration of recording

    # record 5 seconds of sound
    y = sd.rec(int(seconds * sr), samplerate=sr, channels=1)

    # exit loop
    t_end = time.time() + 4.9
    while time.time() < t_end:
        if keyboard.is_pressed('q'):
            print("Q pressed, exiting loop")
            return (0, -1)
    # wait till recording is finished
    sd.wait()

    # save and load latest wavfile
    write('output.wav', sr, y)
    y, sr = librosa.load('output.wav')

    # process audio to correct shape
    spectrogram = process_audio(y, sr)
    spectrograms = np.array(spectrogram)
    spectrograms = spectrograms[None, :, :, :]
    sound = tf.convert_to_tensor(spectrograms, dtype=tf.float32)

    return sound, ext


def plot_loop():
    # load model
    model = keras.models.load_model('Model/model.keras')

    # set up plot
    plt.axis([0, 20, 0, 50])
    plt.title("Live classification", fontsize=20)
    plt.xlabel("time")
    plt.ylabel("hostile-likelyhood")
    yar = []
    xar = []
    x = 1

    while True:
        sound, ext = get_sound()

        if ext == -1:
            plt.close()
            return

        score = model.predict(sound, verbose=0)
        print(score)

        # get last 20 scores
        if x <= 20:
            xar.append(x)
        else:
            yar.pop(0)
        yar.append(score[0][0])
        x += 1

        # plot data
        plt.clf()
        plt.plot(xar, yar)
        plt.pause(0.01)


if __name__ == "__main__":
    plot_loop()

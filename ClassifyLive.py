import os

# Comment if you want to run this on GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import time
import keyboard

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

import sounddevice as sd
from scipy.io.wavfile import write


def process_audio(y, sr):
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = np.concatenate([spectrogram, spectrogram_db])
    # Reshape the 2D spectrogram to have a single channel
    reshaped_spectrogram = np.expand_dims(spectrogram, axis=-1)

    return reshaped_spectrogram / 255


def get_sound():
    ext = 2  # exit flag
    sr = 22050  # Sample rate
    seconds = 1  # Duration of recording

    # record 5 seconds of sound
    y1 = sd.rec(int(seconds * sr), samplerate=sr, channels=1)

    # exit loop
    t_end = time.time() + 0.9
    while time.time() < t_end:
        if keyboard.is_pressed('q'):
            return (0, -1)
    # wait till recording is finished
    sd.wait()

    # save and load latest wavfile
    if os.path.isfile('output5.wav'):
        os.remove("output5.wav")

    if os.path.isfile('output4.wav'):
        os.rename('output4.wav', 'output5.wav')
        y5, sr5 = librosa.load('output5.wav')

    if os.path.isfile('output3.wav'):
        os.rename('output3.wav', 'output4.wav')
        y4, sr4 = librosa.load('output4.wav')

    if os.path.isfile('output2.wav'):
        os.rename('output2.wav', 'output3.wav')
        y3, sr3 = librosa.load('output3.wav')

    if os.path.isfile('output1.wav'):
        os.rename('output1.wav', 'output2.wav')
        y2, sr2 = librosa.load('output2.wav')

    write('output1.wav', sr, y1)
    y1, sr1 = librosa.load('output1.wav')

    if not (os.path.isfile('output5.wav')):
        return (0, 1)

    y = np.concatenate((y5, y4, y3, y2, y1))
    write('output5SEC.wav', sr, y)

    # test should be hostile
    # y, sr = librosa.load('1-9886-A-49.wav')

    # process audio to correct shape
    spectrogram = process_audio(y, sr)
    spectrograms = np.array(spectrogram)
    spectrograms = spectrograms[None, :, :, :]
    sound = tf.convert_to_tensor(spectrograms, dtype=tf.float32)

    return sound, ext


def plot_loop():
    # load model
    model = keras.models.load_model('Model/model.keras')
    model.summary()

    # set up plot
    plt.axis([0, 20, 0, 50])
    plt.title("Live classification", fontsize=20)
    plt.xlabel("time")
    plt.ylabel("hostile-likelyhood")
    yar = []
    xar = []
    x = 1

    while (True):

        sound, ext = get_sound()

        if ext == -1:
            plt.close()
            return

        if ext == 1:
            continue

        score = model.predict(sound, verbose=0)

        # get last 20 scores
        if x <= 20:
            xar.append(x)
        else:
            yar.pop(0)
        yar.append(score[0])
        print(score)
        x += 1

        # plot data
        plt.clf()
        plt.plot(xar, yar)
        plt.pause(0.01)


if __name__ == "__main__":
    plot_loop()

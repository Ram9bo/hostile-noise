import io
import os

#Comment if you want to run this on GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import time
import keyboard
import sys

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from scipy.interpolate import make_interp_spline

import librosa
import librosa.display
import wave
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

    ext = 2 # exit flag
    sr = 22050  # Sample rate
    seconds = 5  # Duration of recording

    #record 5 seconds of sound
    y1 = sd.rec(int(seconds * sr), samplerate=sr, channels=1)

    #exit loop
    t_end = time.time() + 4.9
    while time.time() < t_end:
        if keyboard.is_pressed('q'):
            return(0, -1)
    #wait till recording is finished
    sd.wait()

    #save and load latest wavfile

    write('output.wav', sr, y1)
    y, sr1 = librosa.load('output.wav')
    

    #process audio to correct shape
    spectrogram = process_audio(y, sr)
    spectrograms = np.array(spectrogram)
    spectrograms = spectrograms[None,:,:,:]
    sound = tf.convert_to_tensor(spectrograms, dtype=tf.float32)

    return sound, ext


def plot_loop():

    #load model
    model = keras.models.load_model('Model/model.keras')


    yar = []
    xar = []
    x = 0

    while(True):
    
        #get sound
        sound, ext = get_sound()

        #quit has been called
        if ext == -1:
            plt.close()
            return

        #predict
        score = model.predict(sound, verbose=0)

        #get last 20 scores
        if x <= 20 :
            xar.append(x)
        else:
            yar.pop(0)
        yar.append(score[0][0])
        print(score)
        x +=1

        #plot data
        plt.clf()
        plt.axis([0, 20, 0, 1])
        plt.title("Live classification", fontsize=20)
        plt.xlabel("5-second intervals")
        plt.ylabel("hostile-likelyhood")
        plt.autoscale(False)

        #average line
        plt.axhline(y = np.mean(yar), color = 'r', linestyle = 'dashed')

     
        #trendline
        if len(xar) >=2:
            z = np.polyfit(xar, yar, 1)
            p = np.poly1d(z)
            plt.plot(xar, p(xar), color="purple", linewidth=3, linestyle=":")

        #smooth line
        if len(xar) >=4:
            X_Y_Spline = make_interp_spline(xar, yar)
            X_ = np.linspace(min(xar), max(xar), 500)
            Y_ = X_Y_Spline(X_)
            plt.plot(X_, Y_)
            plt.pause(0.01)

        #starting up
        else:
            plt.plot(xar, yar)
            plt.pause(0.01)
        


if __name__ == "__main__":

    plot_loop()

  
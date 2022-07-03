from tensorflow import keras
import librosa
import numpy as np


MODEL_PATH = 'model.h5'
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
CMDS = ['zero',
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'silence',
        'other']


def load_model():
    return keras.models.load_model(MODEL_PATH)


def predict(model, signal, rate):
    mfcc = librosa.feature.mfcc(y=signal[:rate], sr=rate,
                                n_mfcc=NUM_MFCC,
                                n_fft=N_FFT,
                                hop_length=HOP_LENGTH)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    predictions = model.predict(mfcc)
    predicted_index = np.argmax(predictions)
    predicted_keyword = CMDS[predicted_index]

    return predicted_keyword

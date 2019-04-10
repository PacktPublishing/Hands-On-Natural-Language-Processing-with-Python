from keras.models import load_model
from sklearn.externals import joblib
from constants import *
from processing.proc_audio import from_spectro_to_waveform
import matplotlib.pyplot as plt
import pandas as pd

metadata = pd.read_csv('data/LJSpeech-1.1/metadata.csv',
                       dtype='object', quoting=3, sep='|',
                       header=None)
len_train = int(TRAIN_SET_RATIO * len(metadata))
metadata_testing = metadata.iloc[len_train:]

# load testing data
decoder_input_testing = joblib.load('data/decoder_input_testing.pkl')
mel_spectro_testing = joblib.load('data/mel_spectro_testing.pkl')
spectro_testing = joblib.load('data/spectro_testing.pkl')
text_input_testing = joblib.load('data/text_input_ml_testing.pkl')

# load model
saved_model = load_model('results/model.h5')

predictions = saved_model.predict([text_input_testing, decoder_input_testing])

mel_pred = predictions[0]  # predicted mel spectrogram
mag_pred = predictions[1]  # predicted mag spectrogram


item_index = 0  # pick any index
print('Selected item .wav filename: {}'.format(
    metadata_testing.iloc[item_index][0]))
print('Selected item transcript: {}'.format(
    metadata_testing.iloc[item_index][1]))

predicted_spectro_item = mag_pred[item_index]
predicted_audio_item = from_spectro_to_waveform(predicted_spectro_item, N_FFT,
                                                HOP_LENGTH, WIN_LENGTH,
                                                N_ITER, WINDOW_TYPE,
                                                MAX_DB, REF_DB, PREEMPHASIS)

import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveplot(predicted_audio_item, sr=SAMPLING_RATE)
plt.show()

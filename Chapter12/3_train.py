from constants import *
from sklearn.externals import joblib
from keras.optimizers import Adam
from sklearn.externals import joblib
from model.tacotron_model import get_tacotron_model

# import prepared data
decoder_input_training = joblib.load('data/decoder_input_training.pkl')
mel_spectro_training = joblib.load('data/mel_spectro_training.pkl')
spectro_training = joblib.load('data/spectro_training.pkl')

text_input_training = joblib.load('data/text_input_ml_training.pkl')
vocabulary = joblib.load('data/vocabulary.pkl')

model = get_tacotron_model(N_MEL, r, K1, K2, NB_CHARS_MAX,
                           EMBEDDING_SIZE, MAX_MEL_TIME_LENGTH,
                           MAX_MAG_TIME_LENGTH, N_FFT,
                           vocabulary)

opt = Adam()
model.compile(optimizer=opt,
              loss=['mean_absolute_error', 'mean_absolute_error'])

train_history = model.fit([text_input_training, decoder_input_training],
                          [mel_spectro_training, spectro_training],
                          epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                          verbose=1, validation_split=0.15)


joblib.dump(train_history.history, 'results/training_history.pkl')
model.save('results/model.h5')

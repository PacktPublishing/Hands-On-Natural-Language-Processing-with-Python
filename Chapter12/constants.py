# Audio/Spectral analysis
N_FFT = 1024
PREEMPHASIS = 0.97
SAMPLING_RATE = 16000
FRAME_LENGTH = 0.05  # seconds
FRAME_SHIFT = 0.0125  # seconds
HOP_LENGTH = int(SAMPLING_RATE * FRAME_SHIFT)
WIN_LENGTH = int(SAMPLING_RATE * FRAME_LENGTH)
N_MEL = 80
REF_DB = 20
MAX_DB = 100
r = 5
MAX_MEL_TIME_LENGTH = 200  # Maximum size of the time dimension for a mel spectrogram
MAX_MAG_TIME_LENGTH = 850  # Maximum size of the time dimension for a spectrogram
WINDOW_TYPE='hann'
N_ITER = 50

# Text
NB_CHARS_MAX = 200  # Size of the input text data

# Deep Learning Model
K1 = 16  # Size of the convolution bank in the encoder CBHG
K2 = 8  # Size of the convolution bank in the post processing CBHG
BATCH_SIZE = 32
NB_EPOCHS = 50
EMBEDDING_SIZE = 256

# Other
TRAIN_SET_RATIO = 0.9

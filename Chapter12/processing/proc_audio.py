import os
import librosa
import numpy as np
import copy
from scipy import signal


def get_padded_spectros(filepath, r, preemphasis, n_fft,
                        hop_length, win_length, sampling_rate,
                        n_mel, ref_db, max_db):
    filename = os.path.basename(filepath)
    mel_spectro, spectro = get_spectros(filepath, preemphasis, n_fft,
                                        hop_length, win_length, sampling_rate,
                                        n_mel, ref_db, max_db)
    t = mel_spectro.shape[0]
    nb_paddings = r - (t % r) if t % r != 0 else 0  # for reduction
    mel_spectro = np.pad(mel_spectro,
                         [[0, nb_paddings], [0, 0]],
                         mode="constant")
    spectro = np.pad(spectro,
                     [[0, nb_paddings], [0, 0]],
                     mode="constant")
    return filename, mel_spectro.reshape((-1, n_mel * r)), spectro


def get_spectros(filepath, preemphasis, n_fft,
                 hop_length, win_length,
                 sampling_rate, n_mel,
                 ref_db, max_db):
    waveform, sampling_rate = librosa.load(filepath,
                                           sr=sampling_rate)

    waveform, _ = librosa.effects.trim(waveform)

    # use pre-emphasis to filter out lower frequencies
    waveform = np.append(waveform[0],
                         waveform[1:] - preemphasis * waveform[:-1])

    # compute the stft
    stft_matrix = librosa.stft(y=waveform,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               win_length=win_length)

    # compute magnitude and mel spectrograms
    spectro = np.abs(stft_matrix)

    mel_transform_matrix = librosa.filters.mel(sampling_rate,
                                               n_fft,
                                               n_mel)
    mel_spectro = np.dot(mel_transform_matrix,
                         spectro)

    # Use the decidel scale
    mel_spectro = 20 * np.log10(np.maximum(1e-5, mel_spectro))
    spectro = 20 * np.log10(np.maximum(1e-5, spectro))

    # Normalise the spectrograms
    mel_spectro = np.clip((mel_spectro - ref_db + max_db) / max_db, 1e-8, 1)
    spectro = np.clip((spectro - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose the spectrograms to have the time as first dimension
    # and the frequency as second dimension
    mel_spectro = mel_spectro.T.astype(np.float32)
    spectro = spectro.T.astype(np.float32)

    return mel_spectro, spectro


def get_griffin_lim(spectrogram, n_fft, hop_length,
                    win_length, window_type, n_iter):

    spectro = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        estimated_wav = spectro_inversion(spectro, hop_length,
                                          win_length, window_type)
        est_stft = librosa.stft(estimated_wav, n_fft,
                                hop_length,
                                win_length=win_length)
        phase = est_stft / np.maximum(1e-8, np.abs(est_stft))
        spectro = spectrogram * phase
    estimated_wav = spectro_inversion(spectro, hop_length,
                                      win_length, window_type)
    result = np.real(estimated_wav)

    return result


def spectro_inversion(spectrogram, hop_length, win_length, window_type):
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window=window_type)


def from_spectro_to_waveform(spectro, n_fft, hop_length,
                             win_length, n_iter, window_type,
                             max_db, ref_db, preemphasis):
    # transpose
    spectro = spectro.T

    # de-noramlize
    spectro = (np.clip(spectro, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    spectro = np.power(10.0, spectro * 0.05)

    # wav reconstruction
    waveform = get_griffin_lim(spectro, n_fft, hop_length,
                               win_length,
                               window_type, n_iter)

    # de-preemphasis
    waveform = signal.lfilter([1], [1, -preemphasis], waveform)

    # trim
    waveform, _ = librosa.effects.trim(waveform)

    return waveform.astype(np.float32)

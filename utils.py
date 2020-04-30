import copy

import librosa
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from scipy import signal

import hyperparams as hp


def griffin_lim(spectrogram, n_iter=hp.n_iter):
    x_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        x_t = librosa.istft(x_best,
                            hp.hop_length,
                            win_length=hp.win_length,
                            window="hann")
        est = librosa.stft(x_t,
                           hp.n_fft,
                           hp.hop_length,
                           win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        x_best = spectrogram * phase
    x_t = librosa.istft(x_best,
                        hp.hop_length,
                        win_length=hp.win_length,
                        window="hann")
    y = np.real(x_t)
    return y


def spectrogram2wav(mag):
    # Транспонируем
    mag = mag.T

    # Денормализуем
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # Возвращаемся от децибел к аплитудам
    mag = np.power(10.0, mag * 0.05)

    # Восстанавливаем сигнал
    wav = griffin_lim(mag ** hp.power)

    # De-pre-emphasis фильтр
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def load_spectrograms(path):
    wav, _ = librosa.core.load(path, sr=hp.sr)
    wav, _ = librosa.effects.trim(wav)
    wav = np.append(wav[0], wav[1:] - hp.preemphasis * wav[:-1])
    linear = librosa.stft(y=wav,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)
    mag = np.abs(linear)
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)
    mel = np.dot(mel_basis, mag)

    # Переводим в децибелы
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # Нормализуем
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Транспонируем и приводим к нужным типам
    mel = mel.T.astype(np.float32)
    mag = mag.T.astype(np.float32)

    # Добиваем нулями до правильных размерностей
    t = mel.shape[0]
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

    # Понижаем частоту дискретизации для мел-спектра
    mel = mel[::hp.r, :]

    return mel, mag


def data_load(file_names, data_texts):
    mels = []
    mags = []
    decoder_data = []
    texts = []

    for idx, fname in enumerate(file_names):
        fname += '.npy'
        mel = np.load('mels/' + fname)
        mag = np.load('mags/' + fname)
        t = mel.shape[0]
        nb_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
        mel = np.pad(mel,
                     [[0, nb_paddings], [0, 0]],
                     mode="constant")
        mag = np.pad(mag,
                     [[0, nb_paddings], [0, 0]],
                     mode="constant")
        mel = mel.reshape((-1, hp.n_mels * hp.r))
        decoder_input = tf.concat((tf.zeros_like(mel[:1, :]),
                                   mel[:-1, :]), 0)
        decoder_input = decoder_input[:, -hp.n_mels:]

        mel_padded = np.zeros((hp.max_mel, mel.shape[1]))
        mel_padded[:mel.shape[0], :mel.shape[1]] = mel

        decoder_padded = np.zeros((hp.max_mel, decoder_input.shape[1]))
        decoder_padded[:decoder_input.shape[0], :decoder_input.shape[1]] = decoder_input

        mag_padded = np.zeros((hp.max_mag, mag.shape[1]))
        mag_padded[:mag.shape[0], :mag.shape[1]] = mag

        mels.append(mel_padded)
        mags.append(mag_padded)
        decoder_data.append(decoder_padded)

        text = data_texts[idx].lower()
        text_int = []
        for char in text:
            text_int.append(hp.vocab.find(char))
        texts += [text_int]

    mels = np.array(mels)
    mags = np.array(mags)
    decoder_data = np.array(decoder_data)

    return mels, mags, decoder_data, \
        k.preprocessing.sequence.pad_sequences(texts, maxlen=hp.max_chars,
                                               value=hp.vocab.find('P'))


class DataGen(k.utils.Sequence):

    def __init__(self, batch_size, data):
        names_arr = []
        texts_arr = []
        for i in data:
            name, _, text = i.split('|')
            names_arr.append(name)
            texts_arr.append(text.replace('\r', '').replace('\n', ''))

        self.batch_size = batch_size
        self.texts = texts_arr
        self.filenames = names_arr

    def __len__(self):
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_names = self.filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_texts = self.texts[idx * self.batch_size: (idx + 1) * self.batch_size]
        mels, mags, decoder_data, texts = data_load(batch_names, batch_texts)
        # print(batch_names)
        return [texts, decoder_data], \
               [mels, mags]

import os

import librosa

import hyperparams as hp
from model import build_model
from utils import spectrogram2wav, get_test_data

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

mels, mags, decoder_data, texts = get_test_data()
model = build_model()
model.load_weights('Tacotron (2).h5')
prediction = model.predict([texts, decoder_data])

print(prediction[0][0])
print('""')
print(mels[0])
wav = spectrogram2wav(prediction[1][0])

librosa.output.write_wav('audio.wav', wav, sr=hp.sr)

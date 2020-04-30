import codecs
import os

import librosa
from utils import data_load, spectrogram2wav
import hyperparams as hp
from tensorflow.keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

metadata = codecs.open(hp.path + 'metadata.csv', 'r', 'utf-8').readlines()
name, _, text = metadata[0].split('|')
text = text.replace('\r', '').replace('\n', '')
mels, mags, decoder_data, texts = data_load([name], [text])
model = load_model('Tacotron.h5')
mag = model.predict([texts, decoder_data])[1]

wav = spectrogram2wav(mag[0])

librosa.output.write_wav('audio.wav', wav, sr=hp.sr)

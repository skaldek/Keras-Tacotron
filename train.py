import math
import os

from tensorflow.keras.models import load_model

from model import build_model
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

metadata = codecs.open(hp.path + 'metadata.csv', 'r', 'utf-8').readlines()

data = DataGen(hp.batch_size, metadata)

if os.path.exists('Tacotron.h5'):
    model = load_model('Tacotron.h5')
    print('Loaded model.')
else:
    model = build_model()
    model.compile(loss=['mae', 'mae'], optimizer='adam')

checkpoint = k.callbacks.ModelCheckpoint('Tacotron.h5', verbose=2, monitor='loss')


class EvaluateCallback(k.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mels, mags, decoder_data, texts = get_test_data()
        mag = model.predict([texts, decoder_data])[1]
        wav = spectrogram2wav(mag[0])
        librosa.output.write_wav('audio.wav', wav, sr=hp.sr)


def step_decay(epoch, warmup_steps=4000., steps_per_epoch=data.__len__()):
    if epoch == 0:
        epoch = 1
        steps_per_epoch = 1
    rate = hp.lr * warmup_steps ** 0.5 * tf.minimum(steps_per_epoch * epoch * warmup_steps ** -1.5,
                                                    steps_per_epoch * epoch ** -0.5)
    return rate


lrate = k.callbacks.LearningRateScheduler(step_decay)

model.fit(
    data,
    epochs=hp.epochs,
    callbacks=[checkpoint, lrate, EvaluateCallback()],
    shuffle=True
)

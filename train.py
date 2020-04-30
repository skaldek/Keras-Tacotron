import codecs
import os
from tensorflow.keras.models import load_model
from model import get_model
from utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

metadata = codecs.open(hp.path + 'metadata.csv', 'r', 'utf-8').readlines()

data = DataGen(hp.batch_size, metadata)

if not os.path.exists('Tacotron.h5'):
    model = get_model()
    model.compile(loss=['mean_absolute_error', 'mean_absolute_error'],
                  optimizer=k.optimizers.Adam(hp.lr))
else:
    print('Loaded model')
    model = load_model('Tacotron.h5', custom_objects={'hp': hp})


# model.summary()


checkpoint = k.callbacks.ModelCheckpoint('Tacotron.h5', verbose=2, monitor='loss', save_best_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=1, write_graph=True, write_images=True,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None
)

model.fit(
    data,
    epochs=hp.epochs,
    callbacks=[checkpoint],
    # validation_split=0.1,
    shuffle=True
)

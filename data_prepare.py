import os

from tqdm import tqdm
import pandas as pd
from utils import load_spectrograms
import numpy as np
import hyperparams as hp

data = pd.read_csv(hp.path + 'metadata.csv',
                   header=None, sep='|',
                   names=['Name', 'Text_ints', 'Text'])

if not os.path.exists("mels"):
    os.mkdir("mels")
if not os.path.exists("mags"):
    os.mkdir("mags")

for fname in tqdm(data['Name']):
    fpath = hp.path + fname + '.wav'
    mel, mag = load_spectrograms(fpath)
    np.save("mels/{}".format(fname.replace("wav", "npy")), mel)
    np.save("mags/{}".format(fname.replace("wav", "npy")), mag)

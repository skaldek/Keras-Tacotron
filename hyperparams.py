# Signal processing
sr = 22050  # Sample rate.
n_fft = 2048  # fft points (samples)
frame_shift = 0.0125  # seconds
frame_length = 0.05  # seconds
hop_length = int(sr * frame_shift)  # samples.
win_length = int(sr * frame_length)  # samples.
n_mels = 80  # Number of Mel banks to generate
power = 1.2  # Exponent for amplifying the predicted magnitude
n_iter = 50  # Number of inversion iterations
preemphasis = .97  # or None
max_db = 100
ref_db = 20

prepro = False  # If True - run data_prepare.py first
path = 'RUSLAN/'  # Path
vocab = "P абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz',:;-.!—?«»„“()”–"  # P: Padding

max_mel = 200  # Maximum size of the time dimension for a mel spectrogram
max_mag = 900  # Maximum size of the time dimension for a spectrogram
max_chars = 200

# Model
embed_size = 256
encoder_num_banks = 16
decoder_num_banks = 8
num_highwaynet_blocks = 4
r = 5  # Reduction factor. Paper => 2, 3, 5
dropout_rate = .5

# Training scheme
lr = 0.002  # 0.001  # Initial learning rate.
batch_size = 32
epochs = 500

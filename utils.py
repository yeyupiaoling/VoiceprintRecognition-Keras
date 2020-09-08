import librosa
import mmap
import numpy as np


class AudioData(object):
    def __init__(self, prefix_path):
        self.offset_dict = {}
        for line in open(prefix_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))
        self.fp = open(prefix_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        print('loading label')
        # 获取label
        self.label = {}
        for line in open(prefix_path + '.label', 'rb'):
            key, label = line.split(b'\t')
            self.label[key] = [int(c) for c in label.decode().replace('\n', '').split(',')]
        print('finish loading data:', len(self.label))

    # 获取图像数据
    def get_data(self, key):
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    # 获取图像标签
    def get_label(self, key):
        return self.label.get(key)

    # 获取所有keys
    def get_keys(self):
        return self.label.keys()


# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(path_or_data, sr, mode='train', binary_file=False):
    if not binary_file:
        wav, sr_ret = librosa.load(path_or_data, sr=sr)
        assert sr_ret == sr
    else:
        wav = path_or_data
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
    return linear.T


def load_data(path_or_data, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train', binary_file=False):
    wav = load_wav(path_or_data, sr=sr, mode=mode, binary_file=binary_file)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time - spec_len)
        spec_mag = mag_T[:, randtime:randtime + spec_len]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


# set up multiprocessing
def set_mp(processes=8):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


def get_voxceleb2_datalist(path):
    with open(path, 'r', encoding='utf-8') as f:
        strings = f.readlines()
        audiolist = np.array([string.split()[0] for string in strings])
        labellist = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return audiolist, labellist

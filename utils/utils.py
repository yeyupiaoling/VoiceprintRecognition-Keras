import librosa
import numpy as np


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def load_wav(path, sr, mode='train'):
    wav, sr_ret = librosa.load(path, sr=sr)
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length).T
    mag, _ = librosa.magphase(linear_spect)
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time - spec_len)
        spec_mag = mag_T[:, randtime:randtime + spec_len]
    else:
        spec_mag = mag_T
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


def get_data_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        strings = f.readlines()
        audiolist = np.array([string.split('\t')[0] for string in strings])
        labellist = np.array([int(string.split('\t')[1]) for string in strings])
        f.close()
    return audiolist, labellist

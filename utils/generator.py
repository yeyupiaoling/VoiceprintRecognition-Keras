from tensorflow import keras
import numpy as np

from utils import utils


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, n_classes, batch_size, labels=None, dim=(257, 250, 1), augmentation=True, nfft=512,
                 spec_len=250, win_length=400, sampling_rate=16000, hop_length=160, shuffle=True,normalize=True):
        self.dim = dim
        self.nfft = nfft
        self.sr = sampling_rate
        self.spec_len = spec_len
        self.normalize = normalize
        self.win_length = win_length
        self.hop_length = hop_length

        self.labels = labels
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, indexes):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size,) + self.dim)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, :, :, 0] = utils.load_data(ID, win_length=self.win_length, sr=self.sr, hop_length=self.hop_length,
                                            n_fft=self.nfft, spec_len=self.spec_len)
            # Store class
            y[i] = self.labels[indexes[i]]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

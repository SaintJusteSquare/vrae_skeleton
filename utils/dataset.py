from abc import ABC

import h5py
import glob
import logging
import keras
import numpy as np


class DataGenerator(keras.utils.Sequence, ABC):

    def __init__(self, folder, batch_size, sequence, stage, init_step, shuffle=True):
        self.n_channels = 1
        self.batch_size = batch_size
        self._inputs = "motion"
        self.sequence = sequence
        self.steps = 1
        self.shuffle = shuffle
        print('Searching in {} for files:'.format(folder))
        logging.info('Searching in {} for files:'.format(folder))
        self.list_file = glob.glob('{}/{}*'.format(folder, stage))
        index = []
        for i in range(len(self.list_file)):
            with h5py.File(self.list_file[i], 'r') as f:
                current_lenght = f[self._inputs].shape[0]
                if self.sequence >= current_lenght:
                    logging.error('The lenght of the sequence is larger thant the lenght of the file...')
                    raise ValueError('')
                max_size = current_lenght - (self.sequence + self.steps)
                if not '_dims' in locals():
                    testfile = f[self._inputs]
                    _dims = testfile[0].shape
                    _types = testfile.dtype
                    logging.info('  data label: {} \t dim: {} \t dtype: {}'.format(self._inputs, list(_dims),
                                                                                   _types))
            _index = [[i, x] for x in np.arange(max_size)]
            index += _index
        try:
            self._dims = _dims
        except Exception as e:
            logging.error('Cannot assign dimensions, data not found...')
            raise TypeError(e)
        self._type = _types
        self.idxs = index
        self.init_step = init_step
        self.on_epoch_end()
        logging.info('sequence: {}'.format(self.sequence))
        logging.info('Total of {} files...'.format(len(self.idxs)))

    def __len__(self):
        return int(np.floor(len(self.idxs) / self.batch_size))

    def get_example(self, i):
        # TODO: To change according the train model
        # Currently, fits the train_lstm.
        iDB, iFL = self.idxs[i]
        with h5py.File(self.list_file[iDB], 'r') as f:
            data_label_train = f[self._inputs][iFL: iFL + self.sequence][None, :]
            data_label_test = f[self._inputs][iFL + self.sequence + 1][None, :]
        return data_label_train, data_label_test

    def __getitem__(self, index):
        # TODO: To change according the train model
        # Currently, fits the train_lstm.
        X_seq = np.empty((self.batch_size, self.sequence, *self._dims))
        y_seq = np.empty((self.batch_size, self.sequence, *self._dims))
        for i in range(index, index + self.batch_size):
            example = self.get_example(i)
            t = i - index
            X_seq[t] = example
            y_seq[t] = example
        if self.sequence == 1:
            X_seq = np.squeeze(X_seq)
            y_seq = np.squeeze(y_seq)
        return X_seq, None


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def get_file_data(self, item):
        #item = './data/train/trainf000.h5'
        with h5py.File(item, 'r') as f:
            name = str(np.array(f['song_path']))[1:]
            data = np.array(f['motion'])
            config = np.array(f['position'])
        return name, data, config

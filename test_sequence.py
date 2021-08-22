#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import os
import h5py
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# tf.logging.set_verbosity(tf.logging.DEBUG)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import sys

module_utils = os.path.join(os.getcwd(), 'utils')
sys.path.append(module_utils)
from utils.dataset import DataGenerator

#############################
# Settings
#############################

BATCH_SIZE = 64
time_steps = 100

exp = 'exp'
model_name = os.path.join(exp, 'autoencoder')
data_path = 'data'
configuration = {'file_pos_minmax': 'data/pos_minmax.h5',
                 'normalization': 'interval',
                 'rng_pos': [-0.9, 0.9]}

if not os.path.isdir(exp):
    os.makedirs(exp)

train_path = os.path.join(data_path, 'train')
train_dataset = DataGenerator(train_path, BATCH_SIZE, time_steps, 'train', init_step=1, shuffle=True)

test_path = os.path.join(data_path, 'test')
test_data = DataGenerator(test_path, BATCH_SIZE, time_steps, 'test', init_step=1, shuffle=True)

train_files = os.listdir(train_path)
test_files = os.listdir(test_path)

max_size = 0
max_name = ''
min_size = 100000
min_name = ''
for file in train_files:
    File = os.path.join(train_path, file)
    name, data, config = train_dataset.get_file_data(File)
    if data.shape[0] > max_size:
        max_size = data.shape[0]
        max_name = name
    if data.shape[0] < min_size:
        min_size = data.shape[0]
        min_name = name

print('min dance is ', min_name)
print('min size = ', min_size)
print('max dance is ', max_name)
print('max size = ', max_size)

train_file_path = './data/train/trainf030.h5'
test_file_path  = './data/test/testf000.h5'
name, data, config = train_dataset.get_file_data(train_file_path)
name = name.split('/')[-1]

print('name: ', name)
print('data shape: ', data.shape)
print('data mean: ', np.mean(data))

data = np.reshape(data, (1, data.shape[0], data.shape[1]))
print(data.shape)
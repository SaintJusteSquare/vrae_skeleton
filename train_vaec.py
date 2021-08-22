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
from utils.plot_result import test_draw, draw_image, draw
from motion_transform import reverse_motion_transform

networks = os.path.join(os.getcwd(), 'networks')
sys.path.append(networks)
from networks.CVAE import CVAE

#############################
# Settings
#############################

BATCH_SIZE = 64
time_steps = 1
convolutional = True

exp = 'exp'
model_name = os.path.join(exp, 'convolution')
data_path = 'data'
configuration = {'file_pos_minmax': 'data/pos_minmax.h5',
                 'normalization': 'interval',
                 'rng_pos': [-0.9, 0.9]}

if not os.path.isdir(exp):
    os.makedirs(exp)

################################
# Loading datasets (Train/Test)
################################

train_path = os.path.join(data_path, 'train')
train_dataset = DataGenerator(train_path, BATCH_SIZE, time_steps, 'train', init_step=1, shuffle=True)
original_dim = train_dataset._dims[0]

train_idx = train_dataset.idxs
x_train = np.zeros((len(train_idx), original_dim, time_steps))
if time_steps == 1:
    x_train = np.squeeze(x_train)
for i in range(len(train_idx)):
    x_train[i] = np.squeeze(train_dataset.get_example(i)[0]).T
if convolutional:
    x_train = np.reshape(x_train, (x_train.shape[0], 23, 3, time_steps))

test_path = os.path.join(data_path, 'test')
test_data = DataGenerator(test_path, BATCH_SIZE, time_steps, 'test', init_step=1, shuffle=True)
test_idx = test_data.idxs
x_test = np.zeros((len(test_idx), original_dim, time_steps))
if time_steps == 1:
    x_test = np.squeeze(x_test)
for i in range(len(test_idx)):
    x_test[i] = np.squeeze(test_data.get_example(i)[0]).T
if convolutional:
    x_test = np.reshape(x_test, (x_test.shape[0], 23, 3, time_steps))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train.shape: ', x_train.shape)
print('x_train mean: ', np.mean(x_train))
print('x_test.shape: ', x_test.shape)
print('x_test mean: ', np.mean(x_test))


#############################
# Helper functions
#############################

def plot_result(models, file_name, data_generator, nb_examples=50, batch_size=BATCH_SIZE, model_name="vae",
                untransformed=False):
    # TODO: change paths
    configuration = {'file_pos_minmax': os.path.join(exp, 'data/pos_minmax.h5'),
                     'normalisation': 'interval',
                     'rng_pos': [-0.9, 0.9]}

    vae, encoder, decoder = models

    model_name = os.path.join(exp, model_name)
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data_generator)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2])
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_zlabel("z[2]")
    plt.savefig(filename)
    plt.show()

    name, data, config = data_generator.get_file_data(file_name)
    predicts = vae.predict(data, batch_size)
    name = name[name.rfind('/') + 1:-1]
    h5file = os.path.join(model_name, name)
    if os.path.exists(h5file):
        os.remove(h5file)
    with h5py.File(h5file, 'a') as f:
        f.create_dataset('song_path', data=name)
        f.create_dataset('motion', data=predicts)
        f.create_dataset('position', data=[0, predicts.shape[0]])
    test_draw(h5file, configuration=configuration, speed=25, nb_examples=nb_examples, export_to_file=False, test=False,
              untransformed=untransformed)


def generate_and_save_images(model, epoch, test_input, exp):
    predictions = model.sample(test_input)
    imgs = []
    for i in range(predictions.shape[0]):
        pred = predictions[i, :, :, 0].numpy()
        pred = np.reshape(pred, (1, 69))
        pred = reverse_motion_transform(pred, configuration)
        pred = np.reshape(pred, (23, 3))
        imgs.append(draw_image(pred))
    img1 = np.hstack((imgs[0], imgs[1]))
    img2 = np.hstack((imgs[2], imgs[3]))
    img = np.vstack((img1, img2))

    # tight_layout minimizes the overlap between 2 sub-plots
    folder = os.path.join(exp, 'image_at_epoch_{:04d}.png'.format(epoch))
    cv2.imwrite(folder, np.flip(img, 0))


def reconstruct_sequence(model, test_sequence, exp, export_to_file=True):
    batch_size = test_sequence.shape[0]
    predictions = model.predict_on_batch(test_sequence).numpy()
    predictions = np.reshape(predictions, (batch_size, 69))
    predictions = reverse_motion_transform(predictions, configuration)
    predictions = np.reshape(predictions, (batch_size, 23, 3))
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # opencv3.0
    name = os.path.join(exp, 'test_sequence')
    videoWriter = cv2.VideoWriter(name + '.avi', fourcc, 25, (600, 400))
    draw(predictions, export_to_file=export_to_file, videoWriter_enable=videoWriter)
    videoWriter.release()
    cv2.destroyAllWindows()


def save_plot_model(model, name):
    if not os.path.isdir(name):
        os.makedirs(name)
    tf.keras.utils.plot_model(
        model.inference_net, to_file=os.path.join(name, 'model_inference_net.png'),
        show_shapes=True, show_layer_names=True)
    tf.keras.utils.plot_model(
        model.generative_net, to_file=os.path.join(name, 'model_generative_net.png'),
        show_shapes=True, show_layer_names=True)


def plot_and_save_loss(loss, val_loss, name):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(model_name, 'loss_history.png'))


#############################
# Tensorflow dataset
#############################

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE)


#############################
# Loss
#############################


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_reconstruct = model.decode(z)

    # TODO: has to change the reconstruction term
    reconstruction_term = tf.math.squared_difference(x, x_reconstruct)

    logpx_z = -tf.reduce_sum(reconstruction_term, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def compute_apply_gradients(model, x, optimizer, return_loss=False):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if return_loss:
        return loss


#############################
# Models/training
#############################

epochs = 250
latent_dim = 10
num_examples_to_generate = 4

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([150], [1e-4, 1e-5])
optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

folder_images = os.path.join(exp, 'folder_image')
if not os.path.isdir(folder_images):
    os.makedirs(folder_images)

generate_and_save_images(model, 0, random_vector_for_generation, exp=folder_images)
save_plot_model(model, model_name)

epochs_list = list()
loss_list = list()
val_loss_list = list()
for epoch in range(1, epochs + 1):
    start_time = time.time()
    loss = tf.keras.metrics.Mean()
    for train_x in train_dataset:
        loss(compute_apply_gradients(model, train_x, optimizer1, return_loss=True))

    end_time = time.time()

    if epoch % 1 == 0:
        val_loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            val_loss(compute_loss(model, test_x))
        elbo = -val_loss.result()
        print('Epoch: {}, Test set ELBO: {}, ''time elapse for current epoch {}'.format(epoch,
                                                                                        elbo,
                                                                                        end_time - start_time))
        generate_and_save_images(model, epoch, random_vector_for_generation, exp=folder_images)
        epochs_list.append(epoch)
        val_loss_list.append(val_loss.result())
        loss_list.append(loss.result())

plot_and_save_loss(loss_list, val_loss_list, model_name)

test_sequence_train = tf.convert_to_tensor(x_train[0: 200])
reconstruct_sequence(model, test_sequence_train, exp)

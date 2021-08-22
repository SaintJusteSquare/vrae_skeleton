import tensorflow as tf


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(23, 3, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=6 * 1 * 64, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(6, 1, 64)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="same",
                    activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="same",
                    activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    activation='relu'),
                tf.keras.layers.BatchNormalization(),
                # No activation
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=2, strides=(1, 1), padding='Valid', activation=None),
            ]
        )
        print(self.inference_net.summary())
        print(self.generative_net.summary())

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_tanh=False)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_tanh=False):
        logits = self.generative_net(z)
        if apply_tanh:
            probs = tf.tanh(logits)
            return probs

        return logits

    def predict_on_batch(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_decode = self.decode(z)
        return x_decode

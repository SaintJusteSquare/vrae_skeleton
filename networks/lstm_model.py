import tensorflow as tf


class Lstm_model(tf.keras.Model):
    def __init__(self, timeStep, intermeadiar_dim):
        super(Lstm_model, self).__init__()
        self.timeStep = timeStep
        self.intermediar_dim = intermeadiar_dim
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(timeStep, 69)),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True, activation='tanh'),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True, activation='tanh'),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True, activation='tanh'),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=False, activation='tanh'),
                tf.keras.layers.Dense(69),
            ]
        )

    def call(self, inputs, training=None, mask=None):
        outputs = self.net(inputs)
        return outputs
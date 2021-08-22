import tensorflow as tf


class Seq2seq(tf.keras.Model):
    def __init__(self, timeStep, intermeadiar_dim):
        super(Seq2seq, self).__init__()
        self.timeStep = timeStep
        self.intermediar_dim = intermeadiar_dim
        self.ecoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(timeStep, 69)),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=False, return_state=True),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(timeStep, 69)),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True, activation='tanh'),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True, activation='tanh'),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True, activation='tanh'),
                tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=False, return_state=True,
                                     activation='tanh'),
            ]
        )

    def endode(self, encoder_inputs):
        encoder_outputs, state_h, state_c = self.encoder(encoder_inputs)
        return state_h, state_c
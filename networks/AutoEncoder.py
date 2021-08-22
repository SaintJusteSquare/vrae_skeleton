import tensorflow as tf


class AutoEncoder(tf.keras.Model):
    def __init__(self, timeStep, intermeadiar_dim):
        super(AutoEncoder, self).__init__()
        self.timeStep = timeStep
        self.intermediar_dim = intermeadiar_dim
        self.encoder_inputs = tf.keras.layers.Input(shape=(None, 69))
        self.Dense_encoder_1 = tf.keras.layers.Dense(self.intermediar_dim, activation='relu', name='Dense_encoder_1')
        self.LSTM_encoder_1 = tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True, name='LSTM_encoder_1')
        self.LSTM_encoder_2 = tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True, name='LSTM_encoder_2')
        self.LSTM_encoder_3 = tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True, name='LSTM_encoder_3')
        self.LSTM_encoder_4 = tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True, return_state=True,
                                                   activation='tanh', name='LSTM_encoder_4')

        self.decoder_lstm = tf.keras.layers.LSTM(self.intermediar_dim, return_sequences=True, name='LSTM_decoder')
        self.decoder_dense = tf.keras.layers.Dense(69, activation='tanh', name='Dense_decoder')

        self.encoder = self.make_encoder()
        self.decoder = self.make_decoder()

    def make_encoder(self):
        inputs = self.encoder_inputs
        de1 = self.Dense_encoder_1(inputs)
        le1 = self.LSTM_encoder_1(de1)
        le2 = self.LSTM_encoder_2(le1)
        le3 = self.LSTM_encoder_3(le2)
        encoder_outputs, state_h, state_c = self.LSTM_encoder_4(le3)
        encoder = tf.keras.Model(inputs, [encoder_outputs, state_h, state_c])
        return encoder

    def make_decoder(self):
        decoder_state_input_h = tf.keras.layers.Input(shape=(self.intermediar_dim,))
        decoder_state_input_c = tf.keras.layers.Input(shape=(self.intermediar_dim,))
        decoder_inputs = self.encoder_inputs = tf.keras.layers.Input(shape=(None, self.intermediar_dim))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs = self.decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder = tf.keras.Model([decoder_inputs]+decoder_states_inputs, decoder_outputs)
        return decoder

    def encode(self, inputs):
        de1 = self.Dense_encoder_1(inputs)
        le1 = self.LSTM_encoder_1(de1)
        le2 = self.LSTM_encoder_2(le1)
        le3 = self.LSTM_encoder_3(le2)
        decoder_outputs, state_h, state_c = self.LSTM_encoder_4(le3)
        return decoder_outputs, state_h, state_c

    def call(self, inputs, training=None, mask=None):
        encoder_outputs, state_h, state_c = self.encode(inputs)
        encoder_state = [state_h, state_c]
        decoder_outputs = self.decoder_lstm(encoder_outputs, initial_state=encoder_state)
        decoder_outputs = self.decoder_dense(decoder_outputs)
        return decoder_outputs

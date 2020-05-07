import tensorflow.keras as k
from tensorflow.keras.layers import *
import hyperparams as hp


def highway_layer(value, n_layers, activation="tanh", gate_bias=-3):
    dim = k.backend.int_shape(value)[-1]
    gate_bias_initializer = k.initializers.Constant(gate_bias)
    for i in range(n_layers):
        gate = Dense(units=dim, bias_initializer=gate_bias_initializer)(value)
        gate = Activation("sigmoid")(gate)
        negated_gate = Lambda(
            lambda x: 1.0 - x,
            output_shape=(dim,))(gate)
        transformed = Dense(units=dim)(value)
        transformed = Activation(activation)(transformed)
        transformed_gated = Multiply()([gate, transformed])
        identity_gated = Multiply()([negated_gate, value])
        value = Add()([transformed_gated, identity_gated])
    return value


def get_CBHG_encoder(input_data):
    conv1dbank = get_conv1dbank(hp.encoder_num_banks, input_data)
    conv1dbank = MaxPooling1D(pool_size=2, strides=1,
                              padding='same')(conv1dbank)
    conv1dbank = Conv1D(filters=128, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)
    conv1dbank = Activation('relu')(conv1dbank)
    conv1dbank = Conv1D(filters=128, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)

    residual = Add()([input_data, conv1dbank])

    highway_net = highway_layer(residual, hp.num_highwaynet_blocks, activation='relu')

    cbhg_encoder = Bidirectional(GRU(128, return_sequences=True))(highway_net)

    return cbhg_encoder


def get_CBHG_decoder(input_data):
    conv1dbank = get_conv1dbank(hp.decoder_num_banks, input_data)
    conv1dbank = MaxPooling1D(pool_size=2, strides=1,
                              padding='same')(conv1dbank)
    conv1dbank = Conv1D(filters=256, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)
    conv1dbank = Activation('relu')(conv1dbank)
    conv1dbank = Conv1D(filters=80, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)

    residual = Add()([input_data, conv1dbank])

    highway_net = highway_layer(residual, 4, activation='relu')

    cbhg_decoder = Bidirectional(GRU(128))(highway_net)

    return cbhg_decoder


def get_pre_net(input_data):
    prenet = Dense(256)(input_data)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)
    prenet = Dense(128)(prenet)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)

    return prenet


def get_decoder_RNN_output(input_data):
    rnn1 = GRU(256, return_sequences=True)(input_data)

    inp2 = Add()([input_data, rnn1])
    rnn2 = GRU(256)(inp2)

    decoder_rnn = Add()([inp2, rnn2])

    return decoder_rnn


def get_attention_context(encoder_output, attention_rnn_output):
    attention_input = Concatenate(axis=-1)([encoder_output,
                                            attention_rnn_output])
    e = Dense(10, activation="tanh")(attention_input)
    energies = Dense(1, activation="relu")(e)
    attention_weights = Activation('softmax')(energies)
    context = Dot(axes=1)([attention_weights,
                           encoder_output])

    return context


def get_conv1dbank(K, input_data):
    conv = Conv1D(filters=128, kernel_size=1,
                  strides=1, padding='same')(input_data)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    for k_ in range(2, K + 1):
        conv = Conv1D(filters=128, kernel_size=k_,
                      strides=1, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

    return conv

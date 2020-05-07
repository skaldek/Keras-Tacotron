from modules import *


def build_model():
    # Encoder:
    input_encoder = Input(shape=(hp.max_chars,))

    embedded = Embedding(input_dim=len(hp.vocab),
                         output_dim=hp.embed_size,
                         input_length=hp.max_chars)(input_encoder)
    prenet_encoding = get_pre_net(embedded)

    cbhg_encoding = get_CBHG_encoder(prenet_encoding)

    # Decoder-part1-Prenet:
    input_decoder = Input(shape=(None, hp.n_mels))
    prenet_decoding = get_pre_net(input_decoder)
    attention_rnn_output, state = GRU(256, return_state=True)(prenet_decoding)

    # Attention
    attention_rnn_output_repeated = RepeatVector(
        hp.max_chars)(attention_rnn_output)

    attention_context = get_attention_context(cbhg_encoding,
                                              attention_rnn_output_repeated)

    context_shape1 = int(attention_context.shape[1])
    context_shape2 = int(attention_context.shape[2])
    attention_rnn_output_reshaped = Reshape((context_shape1,
                                             context_shape2))(attention_rnn_output)

    # Decoder-part2:
    input_of_decoder_rnn = concatenate(
        [attention_context, attention_rnn_output_reshaped])
    input_of_decoder_rnn_projected = Dense(256)(input_of_decoder_rnn)

    output_of_decoder_rnn = get_decoder_RNN_output(
        input_of_decoder_rnn_projected)

    mel_hat = Dense(hp.max_mel * hp.n_mels * hp.r)(output_of_decoder_rnn)
    mel_hat = Reshape((hp.max_mel, hp.n_mels * hp.r))(mel_hat)

    mel_hat_last_frame = Lambda(lambda x: x[:, :, -hp.n_mels:])(mel_hat)
    post_process_output = get_CBHG_decoder(mel_hat_last_frame)

    z_hat = Dense(hp.max_mag * (1 + hp.n_fft // 2))(post_process_output)
    z_hat = Reshape((hp.max_mag, (1 + hp.n_fft // 2)))(z_hat)

    model = k.Model(inputs=[input_encoder, input_decoder],
                    outputs=[mel_hat, z_hat])
    return model

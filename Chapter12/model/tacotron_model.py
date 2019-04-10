from keras.layers import Input, Embedding, concatenate, RepeatVector, Dense, Reshape
from keras.models import Model
from .building_blocks import *


def get_tacotron_model(n_mels, r, k1, k2, nb_char_max,
                       embedding_size, mel_time_length,
                       mag_time_length, n_fft,
                       vocabulary):
    # Encoder:
    input_encoder = Input(shape=(nb_char_max,))

    embedded = Embedding(input_dim=len(vocabulary),
                         output_dim=embedding_size,
                         input_length=nb_char_max)(input_encoder)
    prenet_encoding = get_pre_net(embedded)

    cbhg_encoding = get_CBHG_encoder(prenet_encoding,
                                     k1)

    # Decoder-part1-Prenet:
    input_decoder = Input(shape=(None, n_mels))
    prenet_decoding = get_pre_net(input_decoder)
    attention_rnn_output = get_attention_RNN()(prenet_decoding)

    # Attention
    attention_rnn_output_repeated = RepeatVector(
        nb_char_max)(attention_rnn_output)

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

    # mel_hat=TimeDistributed(Dense(n_mels*r))(output_of_decoder_rnn)
    mel_hat = Dense(mel_time_length * n_mels * r)(output_of_decoder_rnn)
    mel_hat_ = Reshape((mel_time_length, n_mels * r))(mel_hat)

    def slice(x):
        return x[:, :, -n_mels:]

    mel_hat_last_frame = Lambda(slice)(mel_hat_)
    post_process_output = get_CBHG_post_process(mel_hat_last_frame,
                                                k2)

    z_hat = Dense(mag_time_length * (1 + n_fft // 2))(post_process_output)
    z_hat_ = Reshape((mag_time_length, (1 + n_fft // 2)))(z_hat)

    model = Model(inputs=[input_encoder, input_decoder],
                  outputs=[mel_hat_, z_hat_])
    return model

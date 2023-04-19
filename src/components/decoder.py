"""
Main decoder script.
"""

# Import dependencies
import tensorflow as tf
import numpy as np

# Import others
from components.decoder_layer import DecoderLayer

# Main decoder class
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, target_vocab_size, max_tokens, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.max_tokens =  max_tokens

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(max_tokens, d_model)

        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights




###########################################
##### MOVE ALL OF THIS TO A TEST FILE #####
###########################################

# # Test
# sample_encoder = TransformerEncoder(1024, vision_transformer_model)
# sample_encoder_output = sample_encoder(resized_image, training=False, mask=None) # call N times // its N parallel times or N sequential?
# sample_decoder = Decoder(num_layers=2, d_model=1024, num_heads=8,
#                          dff=2048, target_vocab_size=8000, max_tokens = 128)
# temp_input = tf.random.uniform((1, 50), dtype=tf.int64, minval=0, maxval=200)

# output, attn = sample_decoder(temp_input,
#                               enc_output=sample_encoder_output,
#                               training=False,
#                               look_ahead_mask=None,
#                               padding_mask=None)

# output.shape, attn['decoder_layer2_block2'].shape

###########################################
##### MOVE ALL OF THIS TO A TEST FILE #####
###########################################
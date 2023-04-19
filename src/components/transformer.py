"""
Main transformer script.
"""

# Import dependencies
import tensorflow as tf
import numpy as np

# Import others
from components.Encoder import Encoder, create_VisionTransformer
from components.decoder import Decoder

# Transformer
class Transformer(tf.keras.Model):
    def __init__(self,*, num_layers, d_model, num_heads, dff,
                target_vocab_size,vision_transformer,  max_tokens, rate=0.1):
        super().__init__()
        self.vision_transformer = vision_transformer
        self.encoder = Encoder(d_model, self.vision_transformer)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            target_vocab_size=target_vocab_size, max_tokens=max_tokens, rate=rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        #print("Transformer call function called")
        inp, tar = inputs
        #print(f"inp: {inp.shape}.     tar: {tar.shape}")
        padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        #print(f"Mask shapes: {padding_mask.shape}.    {look_ahead_mask.shape}")

        enc_output = self.encoder(inp, training, None)  # (batch_size, inp_seq_len, d_model)
        #print("Encoder Works")
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_masks(self, inp, tar):
        # Print the shape of the input tensor
        print("Input tensor shape:", inp.shape) 
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = self.create_padding_mask(inp)
        # Print the shape of the created padding mask
        print("Padding mask shape:", padding_mask.shape)
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask
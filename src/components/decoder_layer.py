"""
Main decoder layer script.
"""

# Import dependencies
import tensorflow as tf
import numpy as np

# Import others
from components.Encoder import MultiHeadAttention

# Decoder Layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        # self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        # self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # print(f"Decoder Layer input x shape: {x.shape}")
        # attn1, attn_weights_block1 = self.mha1(x,x, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)
        # attn1, attn_weights_block1 = self.mha1(x, x, x, attention_mask=look_ahead_mask, return_attention_scores=True)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        # print("attn2 shape: ", attn1.shape)
        # print(f"x shape: {x.shape}")
        out1 = self.layernorm1(attn1 + x)

        #print(f"Encoder outpur (Value and key) shape: {enc_output.shape}")
        #print(f"out1 (Query) shape: {out1.shape}")
        # attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output,  return_attention_scores=True)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, None)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Add, Dense, Dropout, Embedding, GlobalAveragePooling1D, Input, Layer
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from src.logger import logging
from data import data
from components.Embedding import PatchExtractor, PatchEncoder, positional_encoding
from vit_keras import vit
# from data import load_image

vit_model = vit.vit_b32(
        image_size = 224,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        )

new_input = vit_model.input
hidden_layer = vit_model.layers[-2].output
## The New Vision Transformer Model with the required output shapes 
vision_transformer_model = tf.keras.Model(new_input, hidden_layer)

# vision_transformer_model.summary()

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  logging.info("Padding mask is created")
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  logging.info("Look ahead mask is created")
  return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  
  logging.info("Scaled Dot product attention is created")

  return output, attention_weights

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
  print('Attention weights are:')
  print(temp_attn)
  print('Output is:')
  print(temp_out)
  
  
# class MultiHeadAttention(tf.keras.layers.Layer):
#     def __init__(self,*, d_model, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.d_model = d_model

#         assert d_model % self.num_heads == 0

#         self.depth = d_model // self.num_heads

#         self.wq = tf.keras.layers.Dense(d_model)
#         self.wk = tf.keras.layers.Dense(d_model)
#         self.wv = tf.keras.layers.Dense(d_model)

#         self.dense = tf.keras.layers.Dense(d_model)

#     def split_heads(self, x, batch_size):
#         """Split the last dimension into (num_heads, depth).
#         Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
#         """
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
#         return tf.transpose(x, perm=[0, 2, 1, 3])

#     def call(self, v, k, q, mask):
#         batch_size = tf.shape(q)[0]

#         q = self.wq(q)  # (batch_size, seq_len, d_model)
#         k = self.wk(k)  # (batch_size, seq_len, d_model)
#         v = self.wv(v)  # (batch_size, seq_len, d_model)

#         q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
#         k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
#         v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

#         # print(f"Query shape: {q.shape}")
#         # print(f"Key shape: {k.shape}")
#         # print(f"value shape: {v.shape}")

#         # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
#         # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
#         scaled_attention, attention_weights = scaled_dot_product_attention(
#             q, k, v, mask)

#         scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

#         concat_attention = tf.reshape(scaled_attention,
#                                     (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

#         output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
#         logging.info("MultiHead Attention is created")

#         return output, attention_weights

class MLP(Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.dense1 = Dense(hidden_features, activation=tf.nn.gelu)
        self.dense2 = Dense(out_features)
        self.dropout = Dropout(dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y = self.dropout(x)
        return y
    
class Block(Layer):
    def __init__(self, projection_dim, num_heads=4, dropout_rate=0.1):
        super(Block, self).__init__()
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(projection_dim * 2, projection_dim, dropout_rate)

    def call(self, x):
        # Layer normalization 1.
        x1 = self.norm1(x) # encoded_patches
        # Create a multi-head attention layer.
        attention_output = self.attn(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, x]) #encoded_patches
        # Layer normalization 2.
        x3 = self.norm2(x2)
        # MLP.
        x3 = self.mlp(x3)
        # Skip connection 2.
        y = Add()([x3, x2])
        return y
    
class TransformerEncoder(Layer):
    def __init__(self, projection_dim, num_heads=4, num_blocks=8, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.blocks = [Block(projection_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.5)

    def call(self, x):
        # Create a [batch_size, projection_dim] tensor.
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        y = self.dropout(x)
        return y
    
def create_VisionTransformer(num_patches=196, projection_dim=768, input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    # Patch extractor
    patches = PatchExtractor()(inputs)
    # Patch encoder
    patches_embed = PatchEncoder(num_patches, projection_dim)(patches)
    print("Patch encoder",patches_embed.shape)
    # Transformer encoder
    representation = TransformerEncoder(projection_dim)(patches_embed)
    # Create model
    model = Model(inputs=inputs, outputs=representation)
    return model

# class TransformerEncoder(tf.keras.layers.Layer):
    
    
#   def __init__(self, d_model, vision_transformer):
#     super(TransformerEncoder, self).__init__()
#     self.vit = vision_transformer # 12 encoder blocks
#     self.units = d_model
#     self.dense = tf.keras.layers.Dense(self.units, activation=tf.nn.gelu) # FC

#   def call(self, x, training, mask):
#     ## x: (batch, image_size, image_size, 3)
#     x = self.vit(x)
#     x = self.dense(x) 
    
#     logging.info("Encoder is created")
#     return x 
 
class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, vision_transformer):
        super(Encoder, self).__init__()
        self.vit = vision_transformer  # Custom Vision Transformer model
        self.units = d_model
        self.dense = tf.keras.layers.Dense(self.units, activation=tf.nn.gelu)  # FC

    def call(self, x, training, mask):
        ## x: (batch, image_size, image_size, 3)
        x = self.vit(x)
        print(x.shape)
        x = self.dense(x)
        return x
    
vit_model = create_VisionTransformer()
print(vit_model.summary())
 
  # def __init__(self, projection_dim, num_heads=4, num_blocks=12, dropout_rate=0.1):
  #     super(TransformerEncoder, self).__init__()
  #     self.blocks = [Block(projection_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
  #     self.norm = LayerNormalization(epsilon=1e-6)
  #     self.dropout = Dropout(dropout_rate)
  #     self.units = projection_dim
  #     self.dense = tf.keras.layers.Dense(self.units, activation=tf.nn.gelu) # FC

  # def call(self, x, training=None, mask=None):
  #     print("Encoder Input:", x.shape)
  #     x = PatchEncoder()(x)
  #     x+= positional_encoding(x.shape[1], self.units)
  #     # Process the input through the blocks.
  #     for block in self.blocks:
  #         x = block(x, training=training, mask=mask)
  #     x = self.norm(x)
  #     x = self.dropout(x, training=training)
  #     x = self.dense(x)
  #     return x # (1,max_cap_length, projection_dim)
# ### Testing the Encoder
# IMG_PATH = '/home/kishore/workspace/Image-Captioning/data/train2014/COCO_train2014_000000000009.jpg'
# img = plt.imread(IMG_PATH)
# resized_image = tf.image.resize(
#     tf.convert_to_tensor([img]), size=(224, 224)
# )
# print(resized_image.shape)
# sample_encoder = Encoder(1024, vit_model)
# sample_encoder_output = sample_encoder(resized_image, training=False, mask=None) # call N times // its N parallel times or N sequential?
# print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
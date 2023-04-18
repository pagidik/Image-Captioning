import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from data.data import Data
import json 
from components.transformer import Transformer
from components.decoder import Decoder
from components.Encoder import TransformerEncoder

import matplotlib.pyplot as plt
import tensorflow as tf 
import time

with open('parameters/params.json', 'r') as j:
    params = json.load(j)

## Hyperparameters for transformers
vocabulary_size = params["vocab_size"]
num_layers = params["num_layers"]
d_model = params["d_model"]
dff = params["dff"]
num_heads = params["num_heads"]
dropout_rate = params["dropout_rate"]
EPOCHS = params["EPOCHS"]
BATCH_SIZE = params["BATCH_SIZE"]

## Load Dataset
dataset = Data(params)
image_ds , cap_vector = dataset()
# IMG_PATH = '/home/kishore/workspace/Image-Captioning/data/train2014/COCO_train2014_000000000009.jpg'
# resized_image =  plt.imread(IMG_PATH)
print("Dataset Images ", image_ds)
for img in image_ds:
    resized_image = img[0]
    break
resized_image = tf.expand_dims(resized_image, 0)

### Testing the Encoder 
sample_encoder = TransformerEncoder(768)
sample_encoder_output = sample_encoder(resized_image, training=False, mask=None) # call N times // its N parallel times or N sequential?
print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, target_vocab_size=8000, max_tokens = 128)
temp_input = tf.random.uniform((1, 50), dtype=tf.int64, minval=0, maxval=200)

output, attn = sample_decoder(temp_input,
                              enc_output=sample_encoder_output,
                              training=False,
                              look_ahead_mask=None,
                              padding_mask=None)

output.shape, attn['decoder_layer2_block2'].shape

# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#   def __init__(self, d_model, warmup_steps=4000):
#     super(CustomSchedule, self).__init__()

#     self.d_model = d_model
#     self.d_model = tf.cast(self.d_model, tf.float32)

#     self.warmup_steps = warmup_steps

#   def __call__(self, step):
#     arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
#     arg2 = tf.cast(step, tf.float32) * (tf.cast(self.warmup_steps, tf.float32) ** -1.5)
#     return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
# learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)
# temp_learning_rate_schedule = CustomSchedule(d_model)

# plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
# plt.ylabel('Learning Rate')
# plt.xlabel('Train Step')
# plt.show()

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction='none')

# def loss_function(real, pred):
#   mask = tf.math.logical_not(tf.math.equal(real, 0))
#   loss_ = loss_object(real, pred)

#   mask = tf.cast(mask, dtype=loss_.dtype)
#   loss_ *= mask

#   return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


# def accuracy_function(real, pred):
#   accuracies = tf.equal(real, tf.argmax(pred, axis=2))

#   mask = tf.math.logical_not(tf.math.equal(real, 0))
#   accuracies = tf.math.logical_and(mask, accuracies)

#   accuracies = tf.cast(accuracies, dtype=tf.float32)
#   mask = tf.cast(mask, dtype=tf.float32)
#   return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
# transformer = Transformer(
#     num_layers=num_layers,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     target_vocab_size=vocabulary_size,
#     max_tokens = 128,
#     rate=dropout_rate)

# checkpoint_path = './checkpoints/train'

# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)

# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint)
#   print('Latest checkpoint restored!!')

# @tf.function
# def train_step(inp, tar):
#   #print(inp.shape) 
#   #print(tar.shape)
#   tar_inp = tar[:, :-1]
#   tar_real = tar[:, 1:]
#   #print(tar_inp)
#   #print(tar_real)

#   with tf.GradientTape() as tape:
#     predictions, _ = transformer([inp, tar_inp],
#                                  training = True)
#     loss = loss_function(tar_real, predictions)

#   gradients = tape.gradient(loss, transformer.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

#   train_loss(loss)
#   train_accuracy(accuracy_function(tar_real, predictions))

# train_batches = tf.data.Dataset.zip((image_ds, cap_vector))

# train_batches = train_batches.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# for epoch in range(EPOCHS):
#   start = time.time()

#   train_loss.reset_states()
#   train_accuracy.reset_states()

#   # inp -> portuguese, tar -> english
#   for (batch, (inp, tar)) in enumerate(train_batches):
#     #print(inp, tar)
#     train_step(inp, tar)

#     if batch % 50 == 0:
#       print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

#   if (epoch + 1) % 5 == 0:
#     ckpt_save_path = ckpt_manager.save()
#     print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

#   print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

#   print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
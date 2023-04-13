import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from data.data import Data
import json 
from components.transformer import Transformer
import matplotlib.pyplot as plt
import tensorflow as tf 

with open('/home/kishore/workspace/Image-Captioning/parameters/params.json', 'r') as j:
    params = json.load(j)

dataset = Data(params)
# IMG_PATH = '/home/kishore/workspace/Image-Captioning/data/train2014/COCO_train2014_000000000009.jpg'
# resized_image =  plt.imread(IMG_PATH)
print("Dataset Images ", dataset())
for img in dataset():
    resized_image = img
    print(img.shape)
    break
sample_transformer = Transformer(
    num_layers=2, d_model=128, num_heads=4, dff=512, target_vocab_size=8000, max_tokens=128)

temp_input = tf.random.uniform((8, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((1, 36), dtype=tf.int64, minval=0, maxval=200)

fn_out, _ = sample_transformer([resized_image, temp_target], training=False)

fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)

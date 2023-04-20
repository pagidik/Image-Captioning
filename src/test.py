import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from data.data import Data
from components.transformer import Transformer
from vit_keras import vit
import json 
def load_hyperparams(json_file):
    with open(json_file, 'r') as j:
        params = json.load(j)
    return params
def caption_image(image_path, transformer, index_to_word):
    """
    Uses the Transformer passed in the argument to caption the image 
    """
    img = plt.imread(image_path)
    resized_image = tf.image.resize(
        tf.convert_to_tensor([img]), size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    ## Scaling the images
    resized_image = resized_image/255.
    ## Initializing the output arrays 
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, [3])
    output = tf.transpose(output_array.stack())

    for i in tf.range(50):
        output = tf.transpose(output_array.stack())
        predictions, _ = transformer([resized_image, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        output_array = output_array.write(i+1, predicted_id[0])
        if predicted_id == [4]:
            break

    output = tf.transpose(output_array.stack())
    print(index_to_word(output))
    plt.figure(figsize =  (5,  5))
    plt.imshow(resized_image[0])
    plt.axis("off")
    plt.show()

# Set up the Data instance and obtain the test data
params = load_hyperparams('parameters/params.json')

data = Data(params)
image_dataset, cap_vector = data()
test_image_dataset, test_caption_vector = data.get_test_data()
data.mappings()
# Set the IMAGE_SIZE constant to the desired size
IMAGE_SIZE = 224

# Get the index_to_word layer from the Data instance
index_to_word = data.index_to_word
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
# Load the model from the saved checkpoint
checkpoint_path = './checkpoints/train'
ckpt = tf.train.Checkpoint(transformer=Transformer(
    num_layers=params['num_layers'],
    d_model=params['d_model'],
    num_heads=params['num_heads'],
    dff=params['dff'],
    target_vocab_size=params['vocab_size'],
    max_tokens=128,
    vision_transformer=vision_transformer_model,
    rate=params['dropout_rate']),
    optimizer=tf.keras.optimizers.Adam()
)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

transformer = ckpt.transformer
# Caption 15 random images from the test dataset
for i in range(5):
    a = random.randint(0, len(test_image_dataset) - 1)
    img_path = data.image_name_vector_test[a]
    caption_image(img_path, transformer, index_to_word)

'''
Main Train function
'''
import os
import json
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from vit_keras import vit
from data.data import Data
from components.transformer import Transformer
from components.decoder import Decoder
from components.Encoder import Encoder, create_VisionTransformer
from nltk.translate.bleu_score import corpus_bleu
import numpy as np

# Set the GPU configuration
def set_gpu_config():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    physical_devices = tf.config.list_physical_devices()
    print("Physical Devices:", physical_devices)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No compatible GPUs found")

# Load the hyperparameters from JSON file
def load_hyperparams(json_file):
    with open(json_file, 'r') as j:
        params = json.load(j)
    return params

# Create the learning rate scheduler
def create_learning_rate_scheduler(d_model):
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)

            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
            arg2 = tf.cast(step, tf.float32) * (tf.cast(self.warmup_steps, tf.float32) ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    learning_rate = CustomSchedule(d_model)
    return learning_rate

# Plot the learning rate scheduler
def plot_learning_rate_scheduler(learning_rate_schedule):
    plt.plot(learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel('Learning Rate')
    plt.xlabel('Train Step')
    plt.savefig('results/LearningRateScheduler.png')
    plt.show()

# Define the loss function
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# Define the accuracy function
def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

def train_step(inp, tar, transformer, optimizer, loss_object, train_loss, train_accuracy, gradients_values):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions, loss_object)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    gradients_values.extend([grad.numpy().flatten() for grad in gradients])

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))

def main_train(params):
    set_gpu_config()

    ## Load Dataset
    dataset = Data(params)
    image_ds, cap_vector = dataset()
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
    # vit_model = create_vit_model()

    learning_rate = create_learning_rate_scheduler(params['d_model'])
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    transformer = Transformer(
        num_layers=params['num_layers'],
        d_model=params['d_model'],
        num_heads=params['num_heads'],
        dff=params['dff'],
        target_vocab_size=params['vocab_size'],
        max_tokens=128,
        vision_transformer=vision_transformer_model,
        rate=params['dropout_rate'])

    checkpoint_path = './checkpoints/train'

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    image_ds = image_ds.unbatch()
    train_batches = tf.data.Dataset.zip((image_ds, cap_vector))

    train_batches = train_batches.batch(params['BATCH_SIZE']).prefetch(tf.data.AUTOTUNE)

    batch_accuracies = []
    batch_loss = []
    gradients_values = []
    print("Running Epochs...")
    for epoch in range(params['EPOCHS']):
        print("Epoch ", epoch)
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar, transformer, optimizer, loss_object, train_loss, train_accuracy,gradients_values)
            current_accuracy = train_accuracy.result()
            current_loss = train_loss.result()
            batch_accuracies.append(current_accuracy.numpy())
            batch_loss.append(current_loss.numpy())

            if batch % 1 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        if (epoch + 1) % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    # Plot accuracy curve
    plt.plot(batch_accuracies)
    plt.xlabel('Batch Number')
    plt.ylabel('Batch Accuracy')
    plt.title('Batch Accuracy Curve')
    plt.savefig('results/BatchAccuracyCurve.png')
    plt.show()

    # Plot Lsss curve
    plt.plot(batch_accuracies)
    plt.xlabel('Batch Number')
    plt.ylabel('Batch Loss')
    plt.title('Batch Loss Curve')
    plt.savefig('results/BatchLossCurve.png')
    plt.show()

    for layer in transformer.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            plt.hist(layer.kernel.numpy().flatten(), bins=100, alpha=0.5, label='Weights')
        if hasattr(layer, 'bias') and layer.bias is not None:
            plt.hist(layer.bias.numpy().flatten(), bins=100, alpha=0.5, label='Biases')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Weights and Biases Distribution')
    plt.savefig('results/WeightsBiasesDistribution.png')
    plt.show()


    plt.hist(np.concatenate(gradients_values), bins=100)
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.title('Gradients Distribution')
    plt.savefig('results/GradientsDistribution.png')
    plt.show()

    example_input, example_target = list(train_batches.take(1))[0]
    # Get the attention weights from the transformer
    _, attention_weights = transformer([example_input, example_target[:, :-1]], training=False)
    # Plot the attention map for a specific layer and head
    layer = 0
    head = 0
    attention_map = attention_weights[f'decoder_layer{layer + 1}_block1'][0, head].numpy()

    plt.imshow(attention_map)
    plt.colorbar()
    plt.xlabel('Target Position')
    plt.ylabel('Input Position')
    plt.title(f'Attention Map (Layer {layer + 1}, Head {head + 1})')
    plt.savefig(f'results/AttentionMap_Layer{layer + 1}_Head{head + 1}.png')
    plt.show()

if __name__ == "__main__":
    params = load_hyperparams('parameters/params.json')
    main_train(params)


import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import random
import numpy as np
import os
import time
import json
import pickle
from PIL import Image

class Data():
    def __init__(self, params):
        self.data_dir = params['data_dir']
        self.max_cap_length = params['max_cap_length']
        self.vocab_size = params['vocab_size']


    def load_data(self):
        # MS-COCO Image captinoning dataset
        #  Download image annotations
        annotations_folder = self.data_dir + '/annotations/'
        if not os.path.exists(annotations_folder):
            annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                    cache_subdir=os.path.abspath('./data'),
                                                    origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                    extract=True)
            self.annotation_file = os.path.dirname(annotation_zip)+'/data/annotations/captions_train2014.json'
            os.remove(annotation_zip)

        # Dowwnload Images 
        image_folder = self.data_dir + '/train2014/'
        if not os.path.exists(image_folder):
            image_zip = tf.keras.utils.get_file('train2014.zip',
                                                cache_subdir=os.path.abspath('./data'),
                                                origin='http://images.cocodataset.org/zips/train2014.zip',
                                                extract=True)
            self.images_path = os.path.dirname(image_zip) + image_folder
            os.remove(image_zip)

        else:
            self.annotation_file =  annotations_folder + 'captions_train2014.json'
            self.images_path = image_folder
    
    
    def img_to_cap(self):
        # Load annotations json file 
        with open(self.annotation_file,'r') as f:
            self.annotations = json.load(f)
        self.image_path_to_cap = collections.defaultdict(list)

        # Grouping all the captions together for the same image.
        for cap in self.annotations['annotations']:
            caption = f"<start>{cap['caption']}<end>"
            image_path = self.images_path + 'COCO_train2014_' + '%012d.jpg' % (cap['image_id'])
            self.image_path_to_cap[image_path].append(caption)
    
    def path_cap_list(self):
        image_paths = list(self.image_path_to_cap.keys())
        random.shuffle(image_paths)

        self.train_image_paths = image_paths
        
        self.train_captions = []
        self.image_name_vector = []

        for img_path in self.train_image_paths:
            cap_list = self.image_path_to_cap[img_path]
            self.train_captions.extend(cap_list)
            self.image_name_vector.extend([img_path]*len(cap_list))

    def standardize(self, inputs):
        # remove all unnecesary characters in the caption except for '<>' as we want to preserve <start> and <end> tokens.
        inputs = tf.strings.lower(inputs)
        return tf.strings.regex_replace(inputs,
                                        r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")
    
    def tokenizer(self):
        self.caption_dataset = tf.data.Dataset.from_tensor_slices(self.train_captions)

        # Check if vocabulary file exists
        vocab_file = "vocabulary.txt"
        try:
            with open(vocab_file, "r") as f:
                vocabulary = [line.strip() for line in f.readlines()]
            print("Vocabulary loaded from file.")
        except FileNotFoundError:
            text_vectorization = tf.keras.layers.TextVectorization(
                max_tokens=self.vocab_size,
                standardize=self.standardize,
                output_sequence_length=self.max_cap_length)

            text_vectorization.adapt(self.caption_dataset)

            # Save vocabulary to an external file
            vocabulary = text_vectorization.get_vocabulary()
            with open(vocab_file, "w") as f:
                for word in vocabulary:
                    f.write(f"{word}\n")
            print("Vocabulary saved to file.")

        self.tokenizer_object = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size,
            standardize=self.standardize,
            output_sequence_length=self.max_cap_length,
            vocabulary=vocabulary)

        self.caption_vector = self.caption_dataset.map(lambda x: self.tokenizer_object(x))
        return self.caption_vector


    def mappings(self):
        self.word_to_index = tf.keras.layers.StringLookup(
                            mask_token="",
                            vocabulary=self.tokenizer_object.get_vocabulary())
        
        self.index_to_word = tf.keras.layers.StringLookup(
                            mask_token="",
                            vocabulary=self.tokenizer_object.get_vocabulary(),
                            invert=True)
        
    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.keras.layers.Resizing(224, 224)(img)
        return img 
    
    def rescale(self, img):
        return img/255.
    
    def resize(self, img):
        return tf.image.resize(tf.convert_to_tensor([img]), size=(224, 224))

    def transform_image(self):
        # load_image, rescale
        self.image_ds = tf.data.Dataset.from_tensor_slices(self.image_name_vector)
        self.image_ds = self.image_ds.map(self.load_image)
        self.image_ds = self.image_ds.map(self.rescale)
        self.image_ds = self.image_ds.map(self.resize)
        return self.image_ds
    def get_test_data(self):
        # Get the first 10 image paths and captions as test data
        test_image_paths = self.train_image_paths[:10]
        test_captions = []
        test_image_name_vector = []
        for img_path in test_image_paths:
            cap_list = self.image_path_to_cap[img_path]
            test_captions.extend(cap_list)
            test_image_name_vector.extend([img_path] * len(cap_list))

        # Update image_name_vector_test attribute
        self.image_name_vector_test = test_image_name_vector

        # Create test image dataset
        test_image_ds = tf.data.Dataset.from_tensor_slices(test_image_name_vector)
        test_image_ds = test_image_ds.map(self.load_image)
        test_image_ds = test_image_ds.map(self.rescale)
        test_image_ds = test_image_ds.map(self.resize)

        # Create test caption dataset
        test_caption_ds = tf.data.Dataset.from_tensor_slices(test_captions)
        test_caption_vector = test_caption_ds.map(lambda x: self.tokenizer_object(x))

        return test_image_ds, test_caption_vector
    def __call__(self):

        self.load_data()
        self.img_to_cap()
        self.path_cap_list()
        cap_vector = self.tokenizer()
        image_dataset = self.transform_image()
        return image_dataset, cap_vector

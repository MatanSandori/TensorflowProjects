import sys
sys.path.append(".");

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras import utils, layers, applications, Model
from tensorflow.keras.layers.experimental.preprocessing import Resizing, TextVectorization, StringLookup
from tensorflow_helper import helper

import collections
import pathlib
import random
import tqdm
import json
import time
import os

import cv2

helper.SetGpuLimit();
helper.SetSeed();

class dataset:
    def __init__(self, load=100):
        path = "Learning-2/NPL/Seq2Seq/other/ImageToText/Dataset";

        self.images_path = f"{path}/train2014";
        self.captions_path = f"{path}/annotations/captions_train2014.json";

        self.load = load;
    def load_image(self, image_path, size=(128, 128)):
        try:
            image_path = f"{self.images_path}/{image_path}";

            image = tf.io.read_file(image_path);
            image = tf.io.decode_jpeg(image, channels=3);

            image = tf.image.resize(image, size=size);
                
            image = image / 255.0;

            return tf.constant(image);
        except:
            return None;
    def load_captions(self, images_size=(128, 128), use_tqdm=True):
        with open(self.captions_path, "r") as f:
            captions = json.load(f);

        annotations = captions["annotations"];

        images_arr = [];
        captions_arr = [];

        if(self.load is None):
            load = len(annotations);
        else:
            load = self.load;

        for annotation in tqdm.tqdm(
            iterable=annotations[:load],
            desc="loading",
            disable=not(use_tqdm)):

            caption = str(annotation["caption"]);
            image_id = str(annotation["image_id"]);

            zeros_length = 12 - len(image_id);
            string = "0" * zeros_length + image_id;

            image_path = f"COCO_train2014_{string}.jpg";

            image = self.load_image(image_path, images_size);

            if(image == None):
                continue;

            images_arr.append(image);
            captions_arr.append(caption);

        return (images_arr, captions_arr);
    def add_start_end(self, captions):
        result = [];

        for caption in captions:
            caption = str("startsentence ") + str(caption) + str(" endsentence");
            result.append(caption);
        
        return result;
    class tokenize:
        def __init__(self):
            self.vectorization = None;

            self.word_to_index = None;
            self.index_to_word = None;
        def set_captions(self, captions, max_tokens=5000, max_length=50):
            vectorization = TextVectorization(
                max_tokens=max_tokens,
                output_sequence_length=max_length
            );

            vectorization.adapt(captions);

            word_to_index = StringLookup(
                vocabulary=vectorization.get_vocabulary()
            );

            index_to_word = StringLookup(
                vocabulary=vectorization.get_vocabulary(),
                invert=True
            );

            self.vectorization = vectorization;

            self.word_to_index = word_to_index;
            self.index_to_word = index_to_word;

    def as_tfds(self, images, tokens, buffer_size=10000, batch_size=32):
        ds = tf.data.Dataset.from_tensor_slices((images, tokens));
        ds = ds.shuffle(buffer_size);
        ds = ds.batch(batch_size, drop_remainder=True);
        ds = ds.prefetch(tf.data.AUTOTUNE);

        return ds;
    def show(self, images: list, captions: list, sampels=10):
        for _ in range(sampels):
            random_index = random.randint(0, len(images)-1);

            plt.imshow(images[random_index]);
            plt.title(f"{random_index} | {captions[random_index]}");
            plt.show();
    def load_all(self, size=(128, 128), max_tokens=10000, max_length=50, batch_size=32, buffer_size=10000, use_tqdm=True):
        tokanize = self.tokenize();

        (images, captions) = self.load_captions(images_size=(128, 128), use_tqdm=use_tqdm);

        captions = self.add_start_end(captions);

        tokanize.set_captions(
            captions=captions, 
            max_tokens=max_tokens, 
            max_length=max_length);

        tokens = tokanize.vectorization(captions);

        ds = self.as_tfds(images, tokens, buffer_size=buffer_size, batch_size=batch_size);

        if(__name__ == "__main__"):
            self.show(images, captions);

        return ds, tokanize;

if(__name__ == "__main__"):
    timer = helper.Time.Timer();
    time_0 = timer.CreateTimer();

    data = dataset(load=100000);
    ds, tokanize = data.load_all();

    print(tokanize.vectorization.vocabulary_size());

    print(timer.GetTime(time_0));
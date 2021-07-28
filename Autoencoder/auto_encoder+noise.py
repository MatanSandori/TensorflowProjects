import sys
from PIL.Image import Image
from matplotlib import figure
from tensorflow.python.keras.layers.convolutional import Conv
sys.path.append(".");

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_helper import helper
from tensorflow.keras import layers, activations, mixed_precision, metrics, losses, models, Model, applications, optimizers
from tensorflow.keras.layers import Input, Dense, Conv2D, AvgPool2D, MaxPool2D, GlobalAveragePooling2D, GlobalMaxPool2D, Reshape, Flatten, Dropout, BatchNormalization
from tensorflow.keras.datasets import fashion_mnist

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import random
import pathlib
import time
import os

helper.Set();

## Load the dataset

(train_data, _), (test_data, _) = fashion_mnist.load_data();

train_data = train_data.astype("float") / 255.;
test_data = test_data.astype("float") / 255.;

train_data = tf.expand_dims(train_data, axis=-1);
test_data = tf.expand_dims(test_data, axis=-1);

train_noise = 2;
test_noise = 2;

train_data_with_noise = train_data + np.random.uniform(0, train_noise, train_data.shape);
test_data_with_noise = test_data + np.random.uniform(0, test_noise, test_data.shape);

for real, noise in zip(train_data, train_data_with_noise):
    plt.gray();
    plt.imshow(np.squeeze(noise));
    plt.show();
    plt.gray();
    plt.imshow(np.squeeze(real));
    plt.show();
    break;

## Build a model

class EncoderModel(Model):
    def __init__(self):
        super(EncoderModel, self).__init__();
        self.Conv2D_1 = Conv2D(16, 2, activation=activations.relu);
        self.Conv2D_2 = Conv2D(16, 2, activation=activations.relu);
        self.MaxPool = MaxPool2D();
        self.Conv2D_3 = Conv2D(8, 2, activation=activations.relu);
        self.Conv2D_4 = Conv2D(8, 2, activation=activations.relu);
        self.Flatten = Flatten();
        self.dense = Dense(28*28, activation=activations.sigmoid);
        self.outputs = Reshape((28, 28));
    
    def call(self, inputs):
        x = self.Conv2D_1(inputs);
        x = self.Conv2D_2(x);
        x = self.MaxPool(x);
        x = self.Conv2D_3(x);
        x = self.Conv2D_4(x);
        x = self.Flatten(x);
        x = self.dense(x);
        outputs = self.outputs(x);
        return outputs;

model = EncoderModel();

model.compile(
    loss=losses.MeanSquaredError()
);

model.fit(
    train_data_with_noise, train_data, batch_size=32, epochs=10,
    validation_data=(test_data_with_noise, test_data)
);

## Plot predictions

images_preds = model.predict(test_data_with_noise, verbose=1);

def PlotImages(test_data=test_data_with_noise, images_preds=images_preds, loop=1):
    for l in range(loop):
        random_index = random.randint(0, len(test_data) - 1);
        fig, ax = plt.subplots(1, 2, figsize=(10, 10));

        ax[1].imshow(np.squeeze(images_preds[random_index]), cmap=plt.cm.gray);
        ax[1].axis(False);
        ax[1].set_title("Reconstructed");
        ax[0].imshow(np.squeeze(test_data_with_noise[random_index]), cmap=plt.cm.gray);
        ax[0].axis(False);
        ax[0].set_title("Original");
        plt.show();

PlotImages(loop=15);
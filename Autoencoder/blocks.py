import sys
sys.path.append(".");

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_helper import helper
from tensorflow.keras import layers, activations, mixed_precision, metrics, losses, models, Model, applications, optimizers
from tensorflow.keras.layers import Input, Dense, Conv2D, AvgPool2D, MaxPool2D, GlobalAveragePooling2D, GlobalMaxPool2D, Reshape, Flatten, Dropout, BatchNormalization, Conv2DTranspose
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

train_data = train_data.numpy();
test_data = test_data.numpy();

train_data_with_boxes = train_data.copy();
test_data_with_boxes = test_data.copy();

def AddBoxesToImage(data, size=3):
    for image in data:
        for x in range(len(image)):
            for y in range(len(image[0])):
                if(x % size != 0):
                    if(y % size != 0):
                        image[x][y] = 1;

AddBoxesToImage(train_data_with_boxes, 10);
AddBoxesToImage(test_data_with_boxes, 10);

fig, ax = plt.subplots(1, 2);
ax[0].imshow(np.squeeze(train_data_with_boxes[0]), cmap=plt.cm.gray);
ax[1].imshow(np.squeeze(test_data_with_boxes[0]), cmap=plt.cm.gray);
plt.show();

## Build the model

class BoxesRebuilder(Model):
    def __init__(self):
        super(BoxesRebuilder, self).__init__();
        # (28, 28)
        self.conv_1 = Conv2D(8, 4, activation=activations.relu);
        # (24, 24)
        self.conv_2 = Conv2D(8, 4, activation=activations.relu);
        # (20, 20)
        self.conv_3 = Conv2D(8, 4, activation=activations.relu);
        # (16, 16)

        ## Build back the image
        # (16, 16)
        self.conv_transpose_1 = Conv2DTranspose(8, 4, activation=activations.relu);
        # (20, 20)
        self.conv_transpose_2 = Conv2DTranspose(8, 4, activation=activations.relu);
        # (24, 24)
        self.conv_transpose_3 = Conv2DTranspose(8, 4, activation=activations.relu);
        # (28, 28)
        self.outputs = Conv2D(1, 4, activation=activations.sigmoid, padding="same");

    def call(self, inputs):
        x = self.conv_1(inputs);
        x = self.conv_2(x);
        x = self.conv_3(x);

        x = self.conv_transpose_1(x);
        x = self.conv_transpose_2(x);
        x = self.conv_transpose_3(x);
        
        outputs = self.outputs(x);
        return outputs;

model = BoxesRebuilder();

model.compile(loss=losses.MeanSquaredError());

model.fit(
    train_data_with_boxes, train_data, batch_size=32, epochs=10, shuffle=True,
    validation_data=(test_data_with_boxes, test_data)
);

images_preds = model.predict(test_data_with_boxes);

def PlotImages(test_data=test_data_with_boxes, images_preds=images_preds, loop=1):
    for l in range(loop):
        random_index = random.randint(0, len(test_data) - 1);
        fig, ax = plt.subplots(1, 2, figsize=(10, 10));

        ax[0].imshow(np.squeeze(test_data[random_index]), cmap=plt.cm.gray);
        ax[0].set_title("Original");
        ax[0].axis(False);
        ax[1].imshow(np.squeeze(images_preds[random_index]), cmap=plt.cm.gray);
        ax[1].set_title("Reconstructed");
        ax[1].axis(False);
        plt.show();

PlotImages(loop=15);
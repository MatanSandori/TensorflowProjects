import sys
sys.path.append(".");

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_helper import helper
from tensorflow.keras import layers, activations, applications, models, Model, mixed_precision, metrics, losses, optimizers
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, GlobalAveragePooling2D, Conv2D, Dropout, BatchNormalization, Embedding, AvgPool2D, MaxPool2D, Conv2DTranspose, LeakyReLU

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import random
import pathlib
import time
import os

helper.Set();

data_path = pathlib.Path("Learning-2\Audio Recognition\Datasets\Mini-Speech-Commands");
save_path = pathlib.Path("Learning-2/Audio Recognition/Datasets/Saves/Mini-Speach");

class_names = os.listdir(data_path);

train_data = tf.data.experimental.load(os.path.join(save_path, "train_data"));
test_data = tf.data.experimental.load(os.path.join(save_path, "test_data"));

def RescaleData(data):
    images, labels = [], [];

    for image, label in data:
        image = tf.expand_dims(image, axis=-1);
        image = tf.image.resize(image, (32, 32));
        image = tf.cast(tf.math.multiply(image, 10), dtype=tf.float32);
        images.append(image);
        labels.append(label);

    return np.array(images), np.array(labels);

train_images, train_labels = RescaleData(train_data);
test_images, test_labels = RescaleData(test_data);

input_shape = train_images[0].shape;

inputs = Input(shape=input_shape);
x = Conv2D(128, 2, activation=activations.relu)(inputs);
x = Conv2D(128, 2, activation=activations.relu)(x);
x = AvgPool2D()(x);
x = Conv2D(64, 2, activation=activations.relu)(x);
x = Conv2D(64, 2, activation=activations.relu)(x);
x = GlobalAveragePooling2D()(x);
x = Dense(128, activation=activations.relu)(x);
x = Dense(64, activation=activations.relu)(x);
outputs = Dense(len(class_names), activation=activations.softmax)(x);

model = Model(inputs, outputs);

model.compile(
    optimizer=optimizers.RMSprop(),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=metrics.SparseCategoricalAccuracy()
);

history = model.fit(
    train_images, train_labels, epochs=30, batch_size=32,
    validation_data=(test_images, test_labels)
);

helper.Plot.PlotHistory(history);

preds_probs = model.predict(test_images, verbose=1);

preds = preds_probs.argmax(axis=1);

helper.Plot.PlotConfusionMatrix(test_labels, preds, class_names=class_names);
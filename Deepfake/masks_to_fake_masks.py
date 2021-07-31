import sys
from numpy.lib.utils import source

from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import Conv
sys.path.append(".");
sys.path.append("Projects\DeepFake\Scripts");

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_helper import helper
from tensorflow.keras import models, Model, optimizers, losses, metrics, mixed_precision, activations, applications, layers
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Concatenate, GlobalAveragePooling2D, MaxPool2D, AvgPool2D, Flatten, BatchNormalization, Dropout, Reshape, ZeroPadding2D

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pathlib
import random
import tqdm
import time
import os

from dataset_v5 import data

dataset = data(3);
data_a, data_b, data_c = dataset.save.LoadData("dataset_test_5");
fake_masks_o, fake_masks_s = data.extract.Data_b(data_b);
source_masks = data.extract.GetMaskSources(len(fake_masks_s));

for fake_mask_o, fake_mask_s, source_mask in zip(fake_masks_o, fake_masks_s, source_masks):
    fig, axes = plt.subplots(1, 3);
    axes[0].imshow(np.squeeze(fake_mask_o));
    axes[1].imshow(np.squeeze(fake_mask_s));
    axes[2].imshow(np.squeeze(source_mask));
    plt.show();
    break;

def ConvDown(units, kernal, apply_batch_norm=True, strides=2):
    model = models.Sequential();

    model.add(
        Conv2D(
            units, 
            kernal, 
            strides=strides, 
            padding="same"));

    if(apply_batch_norm):
        model.add(BatchNormalization());

    model.add(LeakyReLU());

    return model;

def ConvUp(units, kernal, apply_dropout=True, strides=2):
    model = models.Sequential();

    model.add(
        Conv2DTranspose(
            units,
            kernal,
            strides=strides,
            padding="same",
            activation=activations.relu));

    model.add(BatchNormalization());

    if(apply_dropout):
        model.add(Dropout(0.5));

    return model;

m = 4;

inputs = Input(shape=(256, 256, 1));

inputs = Input(shape=(256, 256, 1));
x = ConvDown(4 * m, 4, False)(inputs);
x = ConvDown(8 * m, 4, True)(x);
x = ConvDown(16 * m, 4, True)(x);
x = ConvDown(32 * m, 4, True)(x);
outputs = ConvDown(64 * m, 4, True)(x);

masks_model = Model(inputs, outputs);

inputs = Input(shape=(256, 256, 1));
x = ConvDown(4 * m, 4, False)(inputs);
x = ConvDown(8 * m, 4, True)(x);
x = ConvDown(16 * m, 4, True)(x);
x = ConvDown(32 * m, 4, True)(x);
outputs = ConvDown(64 * m, 4, True)(x);

source_model = Model(inputs, outputs);

inputs = Concatenate()([
    masks_model.output,
    source_model.output
]);
x = ConvUp(64 * m, 4, True)(inputs);
x = ConvUp(32 * m, 4, True)(x);
x = ConvUp(16 * m, 4, False)(x);
x = ConvUp(8 * m, 4, False)(x);
x = ConvUp(4 * m, 4, False)(x);
outputs = Conv2D(
    7, 4, strides=1, padding="same"
)(x);

model = Model([
    masks_model.input,
    source_model.input
], outputs);

print(model.summary());

model.compile(
    optimizer=optimizers.RMSprop(),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[metrics.SparseCategoricalAccuracy()]
);

model.fit(
    [fake_masks_o,
    source_masks],
    fake_masks_s,
    batch_size=32, epochs=1000, shuffle=True);

save_path = "Projects/DeepFake/Saves/Models/attempt_1";
model.save(f"{save_path}/model_1_2");

preds_probs = model.predict([fake_masks_o, source_masks]);

def CreateMask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1);
    return pred_mask;

for i in range(len(preds_probs)):
    fig, axes = plt.subplots(1, 3);
    axes[0].imshow(tf.squeeze(fake_masks_o[i]));
    axes[1].imshow(tf.squeeze(fake_masks_s[i]));
    axes[2].imshow(tf.squeeze(CreateMask(preds_probs[i])));
    plt.show();
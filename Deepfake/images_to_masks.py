import sys

from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
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

images, fake_masks = data.extract.Data_a(data_a);

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

res_net_50 = applications.ResNet50(include_top=False);
res_net_50.trainable = True;

inputs = Input(shape=(256, 256, 3));
x = res_net_50(inputs, training=True);
x = ConvUp(512, 4, True)(x);
x = ConvUp(256, 4, False)(x);
x = ConvUp(128, 4, False)(x);
x = ConvUp(64, 4, False)(x);
x = ConvUp(32, 4, False)(x);
outputs = Conv2DTranspose(
    6, 4, strides=1, padding="same")(x);

print(outputs.shape);

model = Model(inputs, outputs);

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[metrics.SparseCategoricalAccuracy()]
);

history = model.fit(
    images, fake_masks,
    epochs=400, batch_size=16, shuffle=True);

save_path = "Projects/DeepFake/Saves/Models/attempt_1";
model.save(f"{save_path}/model_1_1");

preds_probs = model.predict(images, batch_size=16, verbose=1);

def CreateMask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1);
    return pred_mask;

for i in range(len(preds_probs)):
    fig, axes = plt.subplots(1, 2);
    axes[0].imshow(tf.squeeze(fake_masks[i]));
    axes[1].imshow(tf.squeeze(CreateMask(preds_probs[i])));
    plt.show();
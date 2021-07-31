import sys
from tensorflow.keras.applications import vgg19

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

masks, targets = data.extract.Data_c(data_c);
sources_images = data.extract.GetSources(len(masks), index=3);

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

res_net_101v2 = applications.ResNet101V2(include_top=False);
res_net_101v2.trainable = True;

res_net_50v2 = applications.ResNet50V2(include_top=False);
res_net_50v2.trainable = True;

inputs = Input(shape=(256, 256, 3));
x = res_net_101v2(inputs, training=True);
x = ConvUp(512, 4, True)(x);
x = ConvUp(256, 4, False)(x);
outputs = ConvUp(256, 4, False)(x);

print(outputs.shape);

target_model = Model(inputs, outputs);

inputs = Input(shape=(256, 256, 1));
x = ConvDown(3, 4, False, 1)(inputs);
x = res_net_50v2(x, training=True);
x = ConvUp(512, 4, True)(x);
x = ConvUp(256, 4, False)(x);
outputs = ConvUp(256, 4, False)(x);

mask_model = Model(inputs, outputs);

print(outputs.shape);

inputs = Concatenate()([
    target_model.output, 
    mask_model.output]);
x = ConvDown(512, 4, True)(inputs);
x = ConvUp(256, 4, True)(x);
x = ConvUp(128, 4, True)(x);
x = ConvUp(64, 4, False)(x);
outputs = Conv2D(3, 4, 1, padding="same")(x);

print(outputs.shape);

model = Model([
    target_model.input,
    mask_model.input
], outputs);

print(model.summary());

model.compile(loss=losses.mae);

model.fit(
    [sources_images,
    masks],
    targets,
    batch_size=4, epochs=400);

save_path = "Projects/DeepFake/Saves/Models/attempt_1";

model.save(f"{save_path}/model_1_3");

preds = model.predict([sources_images, masks], batch_size=12);

for index in range(len(preds)):
    fig, axes = plt.subplots(1, 3);

    plt.gray();
    axes[0].imshow(np.squeeze(np.array((masks[index]), dtype=np.int32)));
    axes[0].axis(False);
    axes[1].imshow(np.squeeze(np.array((targets[index])*255, dtype=np.int32)));
    axes[1].axis(False);
    axes[2].imshow(np.squeeze(np.array((preds[index])*255, dtype=np.int32)));
    axes[2].axis(False);
    plt.show();
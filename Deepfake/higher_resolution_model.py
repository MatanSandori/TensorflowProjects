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

masks, targets = data.extract.Data_c(data_c);
sources_images = data.extract.GetSources(len(masks), 3);

save_path = "Projects/DeepFake/Saves/Models/attempt_1";
model = tf.keras.models.load_model(f"{save_path}/model_1_3");

mask_to_image_preds = model.predict([sources_images, masks], batch_size=12, verbose=1);

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

res_net_50_v2 = applications.ResNet50V2(include_top=False);
res_net_50_v2.trainable = True;

inputs = Input(shape=(256, 256, 3));
x = res_net_50_v2(inputs, training=True);
x = ConvUp(512, 4, True)(x);
x = ConvUp(256, 4, True)(x);
x = ConvUp(128, 4, False)(x);
x = ConvUp(64, 4, False)(x);
x = ConvUp(32, 4, False)(x);
outputs = ConvDown(3, 1, False, 1)(x);

model = Model(inputs, outputs);

model.compile(
    optimizer=optimizers.RMSprop(),
    loss=losses.mae,
    metrics=metrics.mse
);

print(model.summary());

history = model.fit(
    mask_to_image_preds,
    targets,
    epochs=550, batch_size=16
);

helper.Plot.PlotHistory(history);

preds = model.predict(
    mask_to_image_preds,
    batch_size=16, verbose=1);

higher_resolution_save_path = "Projects/DeepFake/Saves/Models/attempt_1/model_1_4";
model.save(higher_resolution_save_path);

for index in range(len(preds)):
    fig, axes = plt.subplots(1, 3);

    axes[0].imshow(np.squeeze(np.array((targets[index]*255), dtype=np.int32)));
    axes[0].axis(False);
    axes[1].imshow(np.squeeze(np.array((mask_to_image_preds[index])*255, dtype=np.int32)));
    axes[1].axis(False);
    axes[2].imshow(np.squeeze(np.array((preds[index])*255, dtype=np.int32)));
    axes[2].axis(False);
    plt.show();
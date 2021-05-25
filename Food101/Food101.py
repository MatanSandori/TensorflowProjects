import tensorflow as tf
import tensorflow_hub as hub

import sys
sys.path.append(".");

from tensorflow_helper import helper
from tensorflow.keras import layers, applications, optimizers, activations, losses, metrics, Model, preprocessing

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.metrics import f1_score, classification_report

import random
import os
import pathlib

image_size = 224;

batch_size = 32;
epochs = 5;

seed = 42;

helper.SetGpuLimit();

data_path = pathlib.Path("Projects\Food101\Data");
image_path = pathlib.Path("Projects\Food101\Data\images");

meta_path = pathlib.Path("Projects\Food101\Data\meta");

checkpoint_callback_path = pathlib.Path("Projects\Food101\Callbacks");

class_names = os.listdir(image_path);

def GetData(path, subset="training", split=0.25):
    return preprocessing.image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        validation_split=split,
        subset=subset if (split is not None) else None,
        seed=seed
);

train_data = GetData(image_path, "training");
test_data = GetData(image_path, "validation");

edited_layer = helper.Image.CreateEditedLayer();

base_line = applications.EfficientNetB0(include_top=False);
base_line.trainable = False;

inputs = layers.Input(shape=(image_size, image_size, 3));
x = edited_layer(inputs);
x = base_line(x);
x = layers.GlobalAveragePooling2D()(x);
outputs = layers.Dense(len(class_names), activation=activations.softmax)(x);

model = Model(inputs, outputs);

#helper.Model.UnfreezeLayers(base_line=base_line);

model.load_weights("Projects\Food101\Callbacks\EfficientNetB0/0.ckpt");

helper.Model.CompileModel(model, loss="sparse");

history = model.fit(
    train_data, batch_size=batch_size, epochs=epochs, steps_per_epoch=len(train_data),
    validation_data=test_data, validation_steps=len(test_data),
    callbacks=[
        helper.Callbacks.CreateCheckpointCallback(checkpoint_callback_path, "EfficientNetB0/1", class_mode="accuracy")]
);

helper.PlotHistory(history);

print(model.evaluate(test_data));

pred_images_path = pathlib.Path("Projects\Food101\Pred_Images");

pred_images = GetData(pred_images_path, split=None);

preds_probs = model.predict(pred_images);

preds = preds_probs.argmax(axis=1);

images = [];

for image in pred_images.unbatch():
    images.append(image);

class_names = class_names[1:];

for i in range(len(images)):
    print(f"pred: {class_names[preds[i]]} \nprob: {preds_probs[i][preds[i]]}\npreds: {preds[i]}\ni: {i}\n");

    plt.matshow(tf.cast(images[i][0], tf.float32) / 255.);
    plt.title(f"pred: {class_names[preds[i]]} | prob: {preds_probs[i][preds[i]]}");
    plt.show();
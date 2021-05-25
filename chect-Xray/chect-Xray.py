import tensorflow as tf
import tensorflow_hub as hub

import sys
sys.path.append(".");

from tensorflow_helper import helper

from tensorflow.keras import layers, activations, models, optimizers, preprocessing, experimental, Model, losses, metrics, applications

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import random
import os
import pathlib

batch_size = 32;

helper.SetGpuLimit();

tf.random.set_seed(42);

data_path = pathlib.Path("Projects/chest xray/chest_xray");
train_data_path = pathlib.Path("Projects/chest xray/chest_xray/train");
test_data_path = pathlib.Path("Projects/chest xray/chest_xray/test");
valid_data_path = pathlib.Path("Projects/chest xray/chest_xray/val");

class_names = os.listdir(train_data_path);

"""
train_data_gen = helper.Image.CreateGen(False);
train_data_gen_edited = helper.Image.CreateGen(True);

test_data_gen = helper.Image.CreateGen(False);
valid_data_gen = helper.Image.CreateGen(False);

train_data = helper.Image.GetDataFromGenDir(train_data_gen, train_data_path);
train_data_edited = helper.Image.GetDataFromGenDir(train_data_gen_edited, train_data_path);

test_data = helper.Image.GetDataFromGenDir(test_data_gen, test_data_path);
valid_data = helper.Image.GetDataFromGenDir(valid_data_gen, valid_data_path);

#helper.Image.PlotRandomImage(train_data_path, class_names=class_names, data=train_data_edited, loop=1);

res_net_v2_50 = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5";
efficient_net_B0 = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1";
mobilenet_v2_100_224 = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5";

Res_net_v2_50(Adam):

    Top epoch(5/5): loss: 0.0592 - categorical_accuracy: 0.9777 - val_loss: 0.5146 - val_categorical_accuracy: 0.8221
    Rate = epoch -> (3/5) || speed -> (slow)

Efficient_net_b0(Adam): 

    Top epoch(3/5): loss: 0.1127 - categorical_accuracy: 0.9607 - val_loss: 0.4546 - val_categorical_accuracy: 0.8157
    Rate = epoch -> (4/5) || speed -> (fast)

Mobile_net_v2_100_200(Adam):

    Top epoch(1/5): loss: 0.3877 - categorical_accuracy: 0.8318 - val_loss: 0.3440 - val_categorical_accuracy: 0.8397
    epoch(5/5): loss: 0.0508 - categorical_accuracy: 0.9842 - val_loss: 0.5835 - val_categorical_accuracy: 0.8173
    Rate = epoch -> (2/5) || speed -> (medium)

Mobile_net_v2_100_200(RMSprop)(Adam):

    (1/5): loss: 0.3082 - categorical_accuracy: 0.8677 - val_loss: 0.3564 - val_categorical_accuracy: 0.8462
"""

"""
depth = 10;

model = models.Sequential([
    hub.KerasLayer(res_net_v2_50, trainable=False),
    layers.Dense(len(class_names), activation=activations.softmax)
]);

model.build([batch_size, 224, 224, 3]);

helper.Model.CompileModel(model, optimizer=optimizers.RMSprop());

history = model.fit(
    train_data, batch_size=batch_size, epochs=5, steps_per_epoch=len(train_data),
    validation_data=test_data, validation_steps=len(test_data)
);

helper.Plot.PlotHistory(history);
"""

def GetData(path):
    return preprocessing.image_dataset_from_directory(
        path,
        image_size=(224,224),
        label_mode="categorical",
        seed=42,
        batch_size=batch_size
    );

train_data = GetData(train_data_path);
test_data = GetData(test_data_path);

#helper.Image.PlotRandomImage(train_data_path, class_names=class_names, loop=10);

data_edited_layer = models.Sequential([
    #layers.experimental.preprocessing.Rescaling(1./255),
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomHeight(0.2),
    layers.experimental.preprocessing.RandomWidth(0.2)
]);

base_model = applications.EfficientNetB0(include_top=False);
base_model.trainable = False;

inputs = layers.Input(shape=(224, 224, 3));
x = data_edited_layer(inputs);
x = base_model(x);
x = layers.GlobalAveragePooling2D()(x);
outputs = layers.Dense(len(class_names), activation=activations.softmax)(x);

model = Model(inputs, outputs);

helper.Model.UnfreezeLayers(20, base_model, True);

helper.Model.CompileModel(model, optimizers.Adam(lr=0.0001));

history = model.fit(
    train_data, batch_size=batch_size, epochs=5, steps_per_epoch=len(train_data),
    validation_data=test_data, validation_steps=int(0.25 * len(test_data))
);

helper.Plot.PlotHistory(history);

model_save_path = pathlib.Path("Projects\chest xray\models");

model.save(model_save_path);

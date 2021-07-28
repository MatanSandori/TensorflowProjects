import sys

from tensorflow.python.keras.layers.preprocessing.text_vectorization import TextVectorization
sys.path.append(".");

from Skimlit_data import data

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_helper import helper
from tensorflow.keras import layers, activations, applications, mixed_precision, metrics, models, Model, optimizers, losses
from tensorflow.keras.layers import Input, Dense, Conv1D, Concatenate, LSTM, GRU, Bidirectional, BatchNormalization, Dropout, GlobalMaxPool1D, GlobalAvgPool1D, MaxPool1D, AvgPool1D, Embedding

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import random
import pathlib
import time
import os

data_obj = data();

data_obj.Load(shuffle=True);

train_df, train_sentances_np, train_labels_np = data_obj.GetTrain();
val_df, val_sentances_np, val_labels_np = data_obj.GetValidation();

class_names = data_obj.GetClassNames();

## Split sentances into characters

train_sentances_split = data_obj.SplitChar(train_sentances_np);
val_sentances_split = data_obj.SplitChar(val_sentances_np);

## Turn mixed presion: on

mixed_precision.set_global_policy("mixed_float16");

## Transfer the data into the right format

max_tokens_char = 300;
max_tokens_sen = 65000;

output_dim = 25;
input_length = 256;

def CreateVecoriazer(tokens):
    return TextVectorization(
        max_tokens=tokens,
        output_mode="int",
        output_sequence_length=output_dim
    );

def CreateEmbedding(tokens):
    return Embedding(
        input_dim=tokens,
        output_dim=output_dim,
        mask_zero=True,
        input_length=input_length
    );

text_vectorization_char = CreateVecoriazer(max_tokens_char);
text_vectorization_char.adapt(train_sentances_split);

text_vectorization_sen = CreateVecoriazer(max_tokens_sen);
text_vectorization_sen.adapt(train_sentances_np);

embedding_char = CreateEmbedding(max_tokens_char);
embedding_sen = CreateEmbedding(max_tokens_sen);

## Set the data

train_sen_char = tf.data.Dataset.from_tensor_slices((train_sentances_np, train_sentances_split));
train_labels = tf.data.Dataset.from_tensor_slices(train_labels_np);

val_sen_char = tf.data.Dataset.from_tensor_slices((val_sentances_np, val_sentances_split));
val_labels = tf.data.Dataset.from_tensor_slices(val_labels_np);

train_data = tf.data.Dataset.zip((train_sen_char, train_labels)).shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE);
val_data = tf.data.Dataset.zip((val_sen_char, val_labels)).batch(32).prefetch(tf.data.AUTOTUNE);

## Build the model

inputs = Input(shape=(1), dtype=tf.string);
x = text_vectorization_sen(inputs);
x = embedding_sen(x);
x = Bidirectional(LSTM(256))(x);
outputs = Dense(128, activation=activations.relu)(x);

model_sen = Model(inputs, outputs);

## Create the 2 model

inputs = Input(shape=(1), dtype=tf.string);
x = text_vectorization_char(inputs);
x = embedding_char(x);
x = Bidirectional(LSTM(256))(x);
outputs = Dense(128, activation=activations.relu)(x);

model_char = Model(inputs, outputs);

## Combine both models

inputs = Concatenate()([model_sen.output, model_char.output]);
x = Dense(128, activation=activations.relu)(inputs);
x = Dense(64, activation=activations.relu)(x);
outputs = Dense(len(class_names), activation=activations.softmax)(x);

model = Model([model_sen.input, model_char.input], outputs);

## Train the model

model.compile(
    optimizer=optimizers.RMSprop(),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=metrics.SparseCategoricalAccuracy()
);

print(model.summary());

history = model.fit(
    train_data, epochs=10, steps_per_epoch=int(0.1 * len(train_data)),
    validation_data=(val_data), validation_steps=int(0.1 * len(val_data))
);

helper.Plot.PlotHistory(history);

"""
10% of the data

RESULTS(shuffle=False):

Epoch 7/10
562/562 [==============================] - 28s 50ms/step - loss: 0.5762 - sparse_categorical_accuracy: 0.7883 - val_loss: 
0.5369 - val_sparse_categorical_accuracy: 0.8045

RESULTS(shuffle=True):

Epoch 10/10
562/562 [==============================] - 26s 47ms/step - loss: 0.5418 - sparse_categorical_accuracy: 0.8019 - val_loss: 
0.4974 - val_sparse_categorical_accuracy: 0.8198

"""
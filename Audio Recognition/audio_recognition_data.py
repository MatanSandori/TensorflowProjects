import sys
sys.path.append(".");

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow_helper import helper
from tensorflow.keras import layers, activations, applications, models, Model, mixed_precision, metrics, losses, optimizers
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, GlobalAveragePooling1D, Conv1D, LSTM, GRU, Bidirectional, Dropout, BatchNormalization, Embedding

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import random
import pathlib
import time
import os

helper.Set();

data_path = pathlib.Path("Learning-2\Audio Recognition\Datasets\Mini-Speech-Commands");

class_names = os.listdir(data_path);

def GetData():
    res = [];
    for class_name in class_names:
        class_str_wav = [];
        class_path = f"{data_path}\{class_name}";
        
        files = os.listdir(class_path);

        for file in files:
            target_path = f"{class_path}\{file}";
            str_wav = tf.io.read_file(target_path);
            audio_tenosr = tf.audio.decode_wav(str_wav);
            class_str_wav.append(audio_tenosr);

        res.append(class_str_wav);

    return res;

data = GetData();

## Plot a random wave

def PlotRandomWave(loop=1):
    for i in range(loop):
        chosen_class_index = random.randint(0, len(class_names) - 1);
        chosen_wave_index = random.randint(0, len(data[chosen_class_index]) - 1);

        print(data[chosen_class_index][chosen_wave_index][0].numpy());

        plt.plot(data[chosen_class_index][chosen_wave_index][0].numpy());
        plt.title(class_names[chosen_class_index]);
        plt.show();

# PlotRandomWave();

## Transer the data into data and labels

full_data = [];
labels = [];
classes = [];

for i in range(len(class_names)):
    for j in range(len(data[i])):
        full_data.append(data[i][j][0].numpy());
        labels.append(i);
        classes.append(class_names[i]);

data_df = pd.DataFrame({"data": full_data, "labels": labels, "classes": classes});
data_df = data_df.sample(frac=1, random_state=42);

data_np = data_df["data"].to_numpy();
data_labels_np = data_df["labels"].to_numpy();

data_tf = [];
for wave in data_np:
    data_tf.append(tf.cast(tf.squeeze(wave), dtype=tf.float32));

def GetSpectrogram(wave):
    padding = tf.zeros([16000] - tf.shape(wave), dtype=tf.float32);
    wave = tf.cast(wave, tf.float32);

    con = tf.concat([wave, padding], axis=0);
    frame = tf.signal.stft(con, 255, 128);
    
    #return tf.cast(frame, dtype=tf.float32)
    frame = tf.abs(frame);
    return frame.numpy();

spectrograms = [];

for wave in data_tf:
    spectrograms.append(GetSpectrogram(wave));

def PlotSpectrogram(spectrogram, ax):
    log = np.log(spectrogram.T);
    heigth = log.shape[0];
    width = log.shape[1];

    x = np.linspace(0, np.size(log.shape), num=width);
    y = range(heigth);

    ax.pcolormesh(x, y, log);

def ShowRandomSpectrogram(loop=1):
    for l in range(loop):
        random_index = random.randint(0, len(data_tf) - 1);

        fig, axes = plt.subplots(2, figsize=(12, 8));
        time_scale = np.arange(data_tf[random_index].shape[0]);

        axes[0].plot(time_scale, data_tf[random_index]);
        axes[0].set_title(data_df["classes"].to_numpy()[random_index]);
        axes[0].set_xlim([0, 16000]);

        PlotSpectrogram(spectrograms[random_index], axes[1]);
        axes[1].set_title("spectrogram");

        plt.show();

ShowRandomSpectrogram(5);

## Split the data

train_size = 0.8;
train_size = int(train_size * len(data_tf));

train_waves, test_waves = data_tf[:train_size], data_tf[train_size:];
train_spectrograms, test_spectrograms = spectrograms[:train_size], spectrograms[train_size:];
train_labels, test_labels = data_labels_np[:train_size], data_labels_np[train_size:];

## Save the data

save_path = pathlib.Path("Learning-2/Audio Recognition/Datasets/Saves/Mini-Speach");

save_path_train_csv = pathlib.Path("Learning-2/Audio Recognition/Datasets/Saves/Mini-Speach/train_data.csv");
save_path_test_csv = pathlib.Path("Learning-2/Audio Recognition/Datasets/Saves/Mini-Speach/test_data.csv");

save_path_train = pathlib.Path("Learning-2/Audio Recognition/Datasets/Saves/Mini-Speach");
save_path_test = pathlib.Path("Learning-2/Audio Recognition/Datasets/Saves/Mini-Speach");

"""
train_data_df = pd.DataFrame({
    "train_waves": train_waves,
    "train_spectrograms": train_spectrograms,
    "train_labels": train_labels
    });

test_data_df = pd.DataFrame({
    "test_waves": test_waves,
    "test_spectrograms": test_spectrograms,
    "test_labels": test_labels
});

train_data_df.to_csv(save_path_train_csv);
test_data_df.to_csv(save_path_test_csv);
"""

train_data = tf.data.Dataset.from_tensor_slices((train_spectrograms, train_labels));
test_data = tf.data.Dataset.from_tensor_slices((test_spectrograms, test_labels));

"""
tf.data.experimental.save(train_data, os.path.join(save_path, "train_data"));
tf.data.experimental.save(test_data, os.path.join(save_path, "test_data"));
"""
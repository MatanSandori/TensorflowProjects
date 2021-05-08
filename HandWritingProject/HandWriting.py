import tensorflow as tf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import idx2numpy

import random
import sys

import itertools

tf.random.set_seed(42);

def PlotRandomImage(data, labels, loop=1, loop_forever=False, y_pred=None, y_prob=None):
    if(loop_forever):
        loop = sys.maxsize;
    for i in range(loop):
        index = random.randint(0, len(data));

        fig, ax = plt.subplots();

        ax.matshow(data[index], cmap=plt.cm.binary);

        def setColor():
            if(y_pred is None):
                return "white";
            if(y_pred[index] == labels[index]):
                return "green";
            return "red";

        def setTitle():
            if(y_pred is None):
                return f"True number: {labels[index]}";
            if(y_prob is None):
                return f"Predict number: {y_pred[index]}"
            return f"Predict number: {y_pred[index]}({int(y_prob[i].max() * 100)}%)";

        ax.set_title(setTitle(), color=setColor());

        print(f"\nPredict number: {y_pred[index]}\n\nTrue Number: {labels[index]}\n");

        if(not(y_pred is None)):
            plt.xlabel(f"True number: {labels[index]}");

        plt.show();

def GetData(path):
    return idx2numpy.convert_from_file(path);

def PlotConfuionMatrix(y_true, y_pred, labels=None, figsize=(10,7), size=17, text_size=5, show_text=True):
    cm = confusion_matrix(y_true, y_pred);

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis];

    n_classes = cm.shape[0];

    fig , ax = plt.subplots(figsize=figsize);

    cax = ax.matshow(cm, cmap=plt.cm.Blues);

    fig.colorbar(cax);

    def SetLabel(labels=labels, n_classes=n_classes):
        if(labels is None):
            return np.arange(n_classes);
        return labels;

    label = SetLabel(labels, n_classes);

    ax.set(
        title="Confusion Matrix",
        xlabel="Predict label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=label,
        yticklabels=label
    );

    ax.xaxis.set_label_position("bottom");
    ax.xaxis.tick_bottom();

    ax.xaxis.label.set_size(size);
    ax.yaxis.label.set_size(size);

    ax.title.set_size(size);

    threshold = (cm.min() + cm.max())/2;

    if(show_text):
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,i,
                f"{cm[i,j]}({cm_norm[i,j]}%)",
                horizontalalignment="center",
                size=text_size,
                color="white" if (cm[i,j] > threshold) else "black"
            );

    plt.show();

plot = False;

file_path = "HandWritingProject";

#Getting the data

train_data = GetData(f"{file_path}/train-images.idx3-ubyte");
train_labels = GetData(f"{file_path}/train-labels.idx1-ubyte");

test_data = GetData(f"{file_path}/t10k-images.idx3-ubyte");
test_labels = GetData(f"{file_path}/t10k-labels.idx1-ubyte");

#Normalizing the data

train_data_norm = train_data / train_data.max();
test_data_norm = test_data / test_data.max();

#Checking the everything works

if(plot):
    PlotRandomImage(train_data_norm, train_labels);
    PlotRandomImage(test_data_norm, test_labels);

"""
print(train_data[0].shape, train_data_norm[0].shape);
(28, 28) - > pixels in image.
"""

#Building a model to learn the data

activations = tf.keras.activations;

def CompileModel(model, optimizer):
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
        );

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(100, activation=activations.relu),
    tf.keras.layers.Dense(10, activation=activations.relu),
    tf.keras.layers.Dense(10, activation=activations.softmax)
]);

#lr = 0.0012 | using callbacks

#callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch / 20));

CompileModel(model, tf.keras.optimizers.Adam(lr=0.0012));

history = model.fit(
    train_data_norm, train_labels, 
    epochs=10,
    validation_data=(test_data_norm, test_labels));

if(plot):
    pd.DataFrame(history.history).plot();

    plt.show();

"""
lrs = 1e-4 * 10 ** (tf.range(50) / 20);

plt.semilogx(lrs, history.history["loss"]);

plt.show();
"""

print(model.evaluate(test_data_norm, test_labels));

y_prob = model.predict(test_data_norm);

y_pred = y_prob.argmax(axis=1);

PlotConfuionMatrix(test_labels, y_pred, show_text=False);

PlotRandomImage(test_data_norm, test_labels, y_pred=y_pred, y_prob=y_prob, loop_forever=True);
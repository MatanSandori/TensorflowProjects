import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

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

def GetPathFromTxt(path):
    value = [];

    with open(f"{meta_path}/{path}.txt", "r") as r:
        value.append(r.readlines());

    return value;

train_path = GetPathFromTxt("train");
test_path = GetPathFromTxt("test");

print(train_path[0][0]);

def SplitArrays(data_path):
    def setArray():
        split_arr = [];
        for i in range(len(data_path[0])):
            split_arr.append(data_path[0][i].split("/"));
        return split_arr;
        
    data_path_name, data_path_images = [], [];

    split_arr = setArray();
    
    for i in range(len(split_arr)):
        data_path_name.append(split_arr[i][0]);
        data_path_images.append(split_arr[i][1]);

    def removeStringFromArray(arr, str="\n"):
        res = [];
        for i in range(len(arr)):
            res.append(arr[i].replace(str, ".jpg"));

        return res;

    data_path_images = removeStringFromArray(data_path_images);

    return data_path_name, data_path_images;

train_path_name, train_path_image = SplitArrays(train_path);
test_path_name, test_path_image = SplitArrays(test_path);

def CreateList(path_image, path_name):
    arr = [];
    for i in range(len(path_image)):
        arr.append(f"{image_path}\{path_name[i]}\{path_image[i]}");
    return arr;

train_path_list = CreateList(train_path_image, train_path_name);
test_path_list = CreateList(test_path_image, test_path_name);

def GetImages(path):
    res = [];

    for i in range(len(path)):
        file = pathlib.Path(path[i]);
        res.append(mpimg.imread(file));

    return res;

train_images = GetImages(train_path_list);
test_images = GetImages(test_path_list);

plt.matshow(train_images[0]);
plt.title(train_path_name[0]);
plt.show();
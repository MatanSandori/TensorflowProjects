import sys
sys.path.append(".");

import tensorflow as tf

from tensorflow_helper import helper
from tensorflow.keras import preprocessing, layers, activations, optimizers, metrics, mixed_precision, applications

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random
import pathlib
import time
import tqdm
import os
import re

class Data():
    def prep(dataset_path, batch_size=16, size=(224, 224)):
        return preprocessing.image_dataset_from_directory(
            dataset_path,
            label_mode="int",
            image_size=size,
            shuffle=True,
            batch_size=batch_size,
            seed=42
        );

    class set:
        def SetPaths(data_path):
            paths = os.listdir(data_path);
            #paths.sort(key=lambda f: int(re.sub('\D', '', f)));
            results_paths = [];

            for path in paths:
                results_paths.append(f"{data_path}/{path}");
            
            return np.array(results_paths);

        def ReadImage(path):
            return np.array(mpimg.imread(path));

        def ReadImages(paths):
            res = [];

            for path in paths:
                res.append(Data.set.ReadImage(path));

            return np.array(res);

        def ResizeImage(image, size=(224, 224)):
            return tf.image.resize(image, size);

    def GetData(batch_size=16, size=(256, 256)):
        dataset_path = pathlib.Path("Projects/Face generation/Dataset");

        data = Data.prep(dataset_path, batch_size=batch_size, size=size);

        data.prefetch(tf.data.AUTOTUNE);
        
        return data;

    def PlotImage(data, take=1, batches=1):
        for image, label in data.take(take):
            for batch in range(batches):
                plt.imshow(tf.cast(image[batch], tf.int32));
                plt.show();

    def GetData_500_600(batch_size=16):
        dataset_path = pathlib.Path("Projects/Face generation/Edited_datasets/500-600");

        data = Data.prep(dataset_path, batch_size=batch_size);

        data.prefetch(tf.data.AUTOTUNE);
        
        return data;

    def GetData_2000_5000(batch_size=16, size=(224, 224)):
        dataset_path = "Projects/Face generation/Edited_datasets/2000-5000";


        data = Data.prep(dataset_path, batch_size=batch_size, size=size);

        data.prefetch(tf.data.AUTOTUNE);
        
        return data;
import sys
sys.path.append(".");
sys.path.append("Projects\DeepFake\Scripts");

import tensorflow as tf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image, ImageDraw, ImageFont

import operator
import pathlib
import random
import time
import cv2
import re
import os

from dataset_v5 import data

def LoadModel(model_path, name="", show_summary=True, save_as_png=False):
    timer = data.timer();
    time_0 = timer.CreateTimer();

    model = tf.keras.models.load_model(model_path);

    if(show_summary):
        print(model.summary());

    if(save_as_png):
        tf.keras.utils.plot_model(model, to_file=f"{name}.png");

    print(f"Time to load {name} model: {timer.GetTime(time_0, convert_to_string=True)}");

    return model;

def CreateMasks(masks):
    def CreateMask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1);
        return pred_mask;

    result = [];
    
    for mask in masks:
        result.append(CreateMask(mask));

    return np.array(result);

dataset = data(3);
_, data_b, data_c = dataset.save.LoadData("dataset_test_5");
_, fake_masks_s = data.extract.Data_b(data_b);
source_masks = data.extract.GetMaskSources(len(fake_masks_s));

paths = data.set.SetPaths("Projects/DeepFake/Dataset/Dataset_v3/Stage 2/0_croped");
images = data.set.ReadImages(paths);

sources_images = data.extract.GetSources(len(images), 3);

save_path = "Projects/DeepFake/Saves/Models/attempt_1";

images_to_masks_save_path = f"{save_path}/model_1_1";
masks_to_fake_masks_save_path = f"{save_path}/model_1_2";
mask_to_image_save_path = f"{save_path}/model_1_3";
higher_resolution_save_path = f"{save_path}/model_1_4";

images_to_masks_model = LoadModel(images_to_masks_save_path, "images_to_masks");
masks_to_fake_masks_model = LoadModel(masks_to_fake_masks_save_path, "masks_to_fake_masks");
fake_masks_to_images_model = LoadModel(mask_to_image_save_path, name="fake_masks_to_images");
higher_resolution_model = LoadModel(higher_resolution_save_path, name="higher_resolution");

images_to_masks_preds = images_to_masks_model.predict(
    images,
    batch_size=16, verbose=1
);

images_to_masks_preds = CreateMasks(images_to_masks_preds);

masks_to_fake_masks_preds = masks_to_fake_masks_model.predict(
    [images_to_masks_preds,
    source_masks],
    batch_size=32, verbose=1
);

masks_to_fake_masks_preds = CreateMasks(masks_to_fake_masks_preds);

mask_to_image_preds = fake_masks_to_images_model.predict(
    [sources_images,
    masks_to_fake_masks_preds],
    batch_size=12, verbose=1);
    
higher_resolution_preds = higher_resolution_model.predict(
    mask_to_image_preds,
    batch_size=16, verbose=1);

all = True;
only_outputs=True;
show = False;
save = True;

all_size=8;
obly_outputs_size=15;

for index in range(len(mask_to_image_preds)):
    if(all):
        fig, axes = plt.subplots(1, 7);

        axes[0].imshow(np.squeeze(np.array((images[index]*255), dtype=np.int32)));
        axes[0].axis(False);
        axes[0].set_title("input_0", fontsize=all_size);
        axes[1].imshow(np.squeeze(np.array((sources_images[index]*255), dtype=np.int32)));
        axes[1].axis(False);
        axes[1].set_title("input_1", fontsize=all_size);
        plt.gray();
        axes[2].imshow(np.squeeze(np.array((images_to_masks_preds[index]), dtype=np.int32)));
        axes[2].axis(False);
        axes[2].set_title("mask", fontsize=all_size);
        plt.gray();
        axes[3].imshow(np.squeeze(np.array((source_masks[index]), dtype=np.int32)));
        axes[3].axis(False);
        axes[3].set_title("source mask", fontsize=all_size);
        plt.gray();
        axes[4].imshow(np.squeeze(np.array((masks_to_fake_masks_preds[index]), dtype=np.int32)));
        axes[4].axis(False);
        axes[4].set_title("fake mask", fontsize=all_size);
        axes[5].imshow(np.squeeze(np.array((mask_to_image_preds[index])*255, dtype=np.int32)));
        axes[5].axis(False);
        axes[5].set_title("fake image", fontsize=all_size);
        axes[6].imshow(np.squeeze(np.array((higher_resolution_preds[index])*255, dtype=np.int32)));
        axes[6].axis(False);
        axes[6].set_title("final output", fontsize=all_size);
        if(save):
            plt.savefig(f"Projects/DeepFake/Saves/Models_outputs/Final Model/all/{index}.png");
        if(show):
            plt.show();
    if(only_outputs):
        fig, axes = plt.subplots(1, 2);

        axes[0].imshow(np.squeeze(np.array((images[index]*255), dtype=np.int32)));
        axes[0].axis(False);
        axes[0].set_title("input_0", fontsize=obly_outputs_size);
        axes[1].imshow(np.squeeze(np.array((higher_resolution_preds[index])*255, dtype=np.int32)));
        axes[1].axis(False);
        axes[1].set_title("final output", fontsize=obly_outputs_size);
        if(save):
            plt.savefig(f"Projects/DeepFake/Saves/Models_outputs/Final Model/only_outputs/{index}.png");
        if(show):
            plt.show();
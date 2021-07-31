import sys
sys.path.append(".");

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
import tqdm
import cv2
import re
import os

class data:
    def __init__(self, dataset_index=2, stage=1):
        ## Image crop -> image = image[650:1500, 750:1700];
        ## Output crop -> frame = frame[20:620,400:875];
        self.dataset_path = f"Projects/DeepFake/Dataset/Dataset_v{dataset_index}";
        self.stage_1_path = f"{self.dataset_path}/Stage 1";
        self.images_path =  f"{self.dataset_path}/Stage {stage}/0_croped";
        self.target_path =  f"{self.dataset_path}/Stage {stage}/1_croped";
        self.mask_path  =  f"{self.dataset_path}/Stage {stage}/2";
        self.fake_mask_o_path = f"{self.dataset_path}/Stage {stage}/3";
        self.fake_mask_s_path = f"{self.dataset_path}/Stage {stage}/4";
        self.source_image_path = f"{self.dataset_path}/Stage 1/1_croped/755.png";

    class extra:
        def ImageToMask(image, dim_4=False):
            if(dim_4):
                image = np.delete(image, 3, 2);

            red = np.array([1, 0, 0]);
            green = np.array([0, 1, 0]);
            blue = np.array([0, 0, 1]);
            white = np.array([1, 1, 1]);
            yellow = np.array([1, 1, 0]);
            cyan = np.array([0, 1, 1]);
            black = np.array([0, 0, 0]);

            final_image = [];

            image = tf.round(image);

            for i in range(len(image)):
                result = [];
                for j in range(len(image[i])):
                    pixel = image[i][j];
                    if(np.array_equal(pixel, black)):
                        result.append(0);
                    elif(np.array_equal(pixel, blue)):
                        result.append(5);
                    elif(np.array_equal(pixel, green)):
                        result.append(4);
                    elif(np.array_equal(pixel, red)):
                        result.append(3);
                    elif(np.array_equal(pixel, yellow)):
                        result.append(2);
                    elif(np.array_equal(pixel, cyan)):
                        result.append(1);
                    elif(np.array_equal(pixel, white)):
                        result.append(3);
                    else:
                        result.append(0);

                final_image.append(result);
            return np.array(final_image);

        def ImagesToMasks(images, name="" ,dim_4=False):
            results = [];

            for image in tqdm.tqdm(images, desc=name):
                results.append(data.extra.ImageToMask(image));

            return np.array(results);

        def MaskToImage(image, dim_4=False):
            if(dim_4):
                image = np.delete(image, 3, 2);

            red = np.array([1, 0, 0]);
            green = np.array([0, 1, 0]);
            blue = np.array([0, 0, 1]);
            white = np.array([1, 1, 1]);
            yellow = np.array([1, 1, 0]);
            cyan = np.array([0, 1, 1]);
            black = np.array([0, 0, 0]);

            final_image = [];

            image = tf.round(image);

            for i in range(len(image)):
                result = [];
                for j in range(len(image[i])):
                    pixel = image[i][j];
                    if(np.array_equal(pixel, 0)):
                        result.append(black);
                    elif(np.array_equal(pixel, 5)):
                        result.append(blue);
                    elif(np.array_equal(pixel, 4)):
                        result.append(green);
                    elif(np.array_equal(pixel, 3)):
                        result.append(red);
                    elif(np.array_equal(pixel, 2)):
                        result.append(yellow);
                    elif(np.array_equal(pixel, 1)):
                        result.append(cyan);
                    elif(np.array_equal(pixel, 3)):
                        result.append(red);
                    else:
                        result.append(0);

                final_image.append(result);
            return np.array(final_image);

    class set:
        def SetPaths(data_path):
            paths = os.listdir(data_path);
            paths.sort(key=lambda f: int(re.sub('\D', '', f)));
            results_paths = [];

            for path in paths:
                results_paths.append(f"{data_path}/{path}");
            
            return np.array(results_paths);

        def ReadImage(path):
            return np.array(mpimg.imread(path));

        def ReadImages(paths):
            res = [];

            for path in paths:
                res.append(data.set.ReadImage(path));

            return np.array(res);

        def ResizeImage(image, size=(256, 256)):
            return tf.image.resize(image, size);

        def CropImage(image, size=(330, 120)):
            def CropND(img, bounding):
                start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding));
                end = tuple(map(operator.add, start, bounding));
                slices = tuple(map(slice, start, end));
                return np.array(img[slices], dtype=np.float32);

            res = CropND(image, size);
            return res;

        def CropWithoutBound(images, pt1=[100, 100], pt2=[300, 300]):
            res = [];

            for image in images:
                res.append(image[pt1[0]:pt1[1], pt2[0]:pt2[1]]);

            return np.array(res);

        def CreateMask(pred_mask):
            pred_mask = tf.argmax(pred_mask, axis=-1);
            return pred_mask;

        def CreateMasks(preds_masks):
            results = [];

            for pred_mask in preds_masks:
                results.append(data.set.CreateMask(pred_mask));

            return np.array(results);

    class get:
        def GetData(data_index=2, stage=1, lim=10, data_c_lim=None):
            if(data_c_lim is None):
                data_c_lim = lim;
            dataset = data(data_index, stage);

            images_paths = dataset.set.SetPaths(dataset.images_path)[:lim];
            targets_paths = dataset.set.SetPaths(dataset.target_path)[:data_c_lim];

            masks_paths = dataset.set.SetPaths(dataset.mask_path)[:data_c_lim];
            fake_masks_o_paths = dataset.set.SetPaths(dataset.fake_mask_o_path)[:lim];
            fake_masks_s_paths = dataset.set.SetPaths(dataset.fake_mask_s_path)[:lim];

            images = dataset.set.ReadImages(images_paths);
            targets = dataset.set.ReadImages(targets_paths);

            masks = dataset.set.ReadImages(masks_paths);
            fake_masks_o = dataset.set.ReadImages(fake_masks_o_paths);
            fake_masks_s = dataset.set.ReadImages(fake_masks_s_paths);

            masks = data.extra.ImagesToMasks(masks, "masks");
            fake_masks_o = data.extra.ImagesToMasks(fake_masks_o, "fake_masks_o");
            fake_masks_s = data.extra.ImagesToMasks(fake_masks_s, "fake_masks_s");

            return images, fake_masks_o, fake_masks_s, masks, targets;

    class save:
        def SaveData(images, fake_masks_o, fake_masks_s, masks, targets, name="dataset_test_2"):
            save_path = f"Projects/DeepFake/Saves/Datasets/{name}";

            images = tf.data.Dataset.from_tensor_slices(images);
            fake_masks_o = tf.data.Dataset.from_tensor_slices(fake_masks_o);
            fake_masks_s = tf.data.Dataset.from_tensor_slices(fake_masks_s);

            masks = tf.data.Dataset.from_tensor_slices(masks);
            targets = tf.data.Dataset.from_tensor_slices(targets);

            data_a = tf.data.Dataset.zip((images, fake_masks_o));
            data_b = tf.data.Dataset.zip((fake_masks_o, fake_masks_s));
            data_c = tf.data.Dataset.zip((masks, targets));

            tf.data.experimental.save(data_a, f"{save_path}/data_a");
            tf.data.experimental.save(data_b, f"{save_path}/data_b");
            tf.data.experimental.save(data_c, f"{save_path}/data_c");

        def LoadData(name="dataset_test_2"):
            save_path = f"Projects/DeepFake/Saves/Datasets/{name}";
            
            data_a  = tf.data.experimental.load(f"{save_path}/data_a");
            data_b = tf.data.experimental.load(f"{save_path}/data_b");
            data_c = tf.data.experimental.load(f"{save_path}/data_c");

            return data_a, data_b, data_c;

    class extract:
        def Data_a(data_a):
            images, fake_masks = [], [];

            for image, fake_mask in data_a:
                fake_mask = np.expand_dims(fake_mask, axis=-1);

                images.append(image);
                fake_masks.append(fake_mask);

            return np.array(images), np.array(fake_masks);

        def Data_b(data_b):
            fake_masks_o, fake_masks_s = [], [];

            for fake_mask_o, fake_mask_s in data_b:
                fake_mask_o = np.expand_dims(fake_mask_o, axis=-1);
                fake_mask_s = np.expand_dims(fake_mask_s, axis=-1);

                fake_masks_o.append(fake_mask_o);
                fake_masks_s.append(fake_mask_s);

            return np.array(fake_masks_o), np.array(fake_masks_s);

        def Data_c(data_c):
            masks, targets = [], [];

            for mask, target in data_c:
                mask = np.expand_dims(mask, axis=-1);

                masks.append(mask);
                targets.append(target);

            return np.array(masks), np.array(targets);

        def GetSources(length=None, index=2):
            result = [];

            dataset = data(index);
            source_image = mpimg.imread(dataset.source_image_path);

            for i in range(length):
                result.append(source_image);
            
            return np.array(result);

        def GetMaskSources(length):
            result = [];

            source_image = mpimg.imread("Projects/DeepFake/Dataset/Dataset_v3/Stage 3/755.png");
            source_image = data.extra.ImageToMask(source_image);

            for i in range(length):
                result.append(source_image);
            
            return np.array(result);

        def ExtractFromDataC(data_c, length):
            results = [];
            mask_0 = None;

            for mask, _ in data_c:
                mask_0 = np.expand_dims(mask, axis=-1);
                break;
            
            for i in range(length):
                results.append(mask_0);

            return np.array(results);

    class build:
        def VideoToFrames(video_path, save_path=None, save_rate=1, start_index=0):
            cap = cv2.VideoCapture(video_path);

            i = start_index;

            while True:
                try:
                    rec, frame = cap.read();
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

                    if(i % save_rate == 0):
                        frame = frame / 255.;
                        frame = data.set.ResizeImage(frame, size=(256, 256));

                        if(save_path is None):
                            plt.imshow(np.squeeze(frame));
                            plt.show();
                        else:
                            frame = np.array(frame*255, dtype=np.uint8);
                            image = Image.fromarray(frame);
                            image.save(f"{save_path}/{i}.png");
                    i+=1;
                except:
                    break;

    class timer:
        def __init__(self):
            self.times = [];
            
        def CreateTimer(self):
            start_time = time.time();
            self.times.append(start_time);
            return len(self.times)-1;
            
        def GetTime(self, index=0, round_time=True, convert_to_string=False):
            current_time = time.time();
            result_time = current_time - self.times[index];
            if(round_time):
                result_time = round(result_time, 2);
            if(convert_to_string):
                result_time = f"{result_time}s";
            return result_time;
            
        def Sleep(self, sec=None):
            time.sleep(sec);

save = False;

if(save):
    dataset = data(3, 2);

    images, fake_masks_o, fake_masks_s, masks, targets = dataset.get.GetData(3, 2, lim=23, data_c_lim=117);

    dataset.save.SaveData(
        images, fake_masks_o, fake_masks_s, masks, targets,
        name="dataset_test_5");

    data_a, data_b, data_c = dataset.save.LoadData("dataset_test_5");

    print(data_a, data_b, data_c);

    for image, fake_mask in data_a.take(1):
        print(fake_mask.shape);
        plt.imshow(np.squeeze(fake_mask));
        plt.gray();
        plt.show();

"""
vid_path = "Projects/DeepFake/Dataset/Dataset_v0/Stage 0/output.mp4";

cap = cv2.VideoCapture(vid_path);

i = 0;
count = 17;

while True:
    try:
        rec, frame = cap.read();
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

        if(i >= count):
            if(i % 5 == 0):
                save_path = "Projects/DeepFake/Dataset/Dataset_v3/Stage 1/1_croped";
                frame = frame[20:620,350:900];
                frame = frame / 255.;
                frame = data.set.ResizeImage(frame, size=(256, 256));
                image = np.array(frame*255, dtype=np.uint8);
                image = Image.fromarray(image);
                image.save(f"{save_path}/{count}.png");
            
            count+=1;

        i += 1;
    except:
        break;
"""
#vid_path = "Projects/DeepFake/Dataset/Dataset_v3/Stage 2/0_o/Untitled.mp4";
#save_path = "Projects/DeepFake/Dataset/Dataset_v3/Stage 2/0_croped";

#data.build.VideoToFrames(vid_path, save_path, save_rate=5, start_index=0);
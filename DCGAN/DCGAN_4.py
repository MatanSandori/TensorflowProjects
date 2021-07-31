import sys

from tensorflow.python.ops.gen_math_ops import Greater, atan
sys.path.append(".");

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_helper import helper
from tensorflow.keras import layers, activations, mixed_precision, metrics, losses, models, Model, applications, optimizers
from tensorflow.keras.layers import Input, Dense, Conv2D, AvgPool2D, MaxPool2D, GlobalAveragePooling2D, GlobalMaxPool2D, Reshape, Flatten, Dropout, BatchNormalization, Conv2DTranspose, LeakyReLU
from tensorflow.keras.datasets import mnist

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

import random
import pathlib
import time
import tqdm
import os

from faces_dataset import Data as faces_data_set

batch_size = 100;
rand = 100;

helper.SetGpuLimit();
data = faces_data_set.GetData_2000_5000(batch_size=batch_size, size=(256, 256));

gen_optimizer = optimizers.Adagrad();
dis_optimizer = optimizers.Adam(learning_rate=1e-4, beta_1=0.9);

def GeneratorModel(m=1, k=4):
    inputs = Input(shape=(rand));
    x = Dense(4*4*(int(512*m)), use_bias=False)(inputs);
    x = Reshape((4, 4, int(512*m)))(x);
    ## (None, 4, 4, 512)
    x = Conv2DTranspose(int(256*m), k, strides=2, padding="same")(x);
    x = LeakyReLU(alpha=0.2)(x);
    ## (None, 8, 8, 256)
    x = Conv2DTranspose(int(128*m), k, strides=2, padding="same")(x);
    x = LeakyReLU(alpha=0.2)(x);
    ## (None, 16, 16, 128)
    x = Conv2DTranspose(int(64*m), k, strides=2, padding="same")(x);
    x = LeakyReLU(alpha=0.2)(x);
    ## (None, 32, 32, 64)
    x = Conv2DTranspose(int(32*m), k, strides=2, padding="same")(x);
    x = LeakyReLU(alpha=0.2)(x);
    ## (None, 64, 64, 32)
    x = Conv2DTranspose(int(16*m), k, strides=2, padding="same")(x);
    x = LeakyReLU(alpha=0.2)(x);
    ## (None, 128, 128, 16)
    outputs = Conv2DTranspose(3, k, strides=2, padding="same", activation=activations.sigmoid)(x);
    ## (None, 256, 256, 3)
    print(outputs.shape);

    return Model(inputs, outputs);

def DisclaimerModel(m=1, k=4):
    inputs = Input(shape=(256, 256, 3));
    x = layers.experimental.preprocessing.Rescaling(1./255)(inputs);
    x = Conv2D(int(16*m), k, strides=2, padding="same")(x);
    x = LeakyReLU(alpha=0.2)(x);
    ## (None, 128, 128, 16)
    x = Conv2D(int(32*m), k, strides=2, padding="same")(x);
    x = LeakyReLU(alpha=0.2)(x);
    ## (None, 64, 64, 64)
    x = Conv2D(int(64*m), k, strides=2, padding="same")(x);
    x = LeakyReLU(alpha=0.2)(x);
    ## (None, 32, 32, 64)
    x = Conv2D(int(128*m), k, strides=2, padding="same")(x);
    x = LeakyReLU(alpha=0.2)(x);
    ## (None, 8, 8, 128)
    x = Conv2D(int(256*m), k, strides=2, padding="same")(x);
    x = LeakyReLU(alpha=0.2)(x);
    ## (None, 4, 4, 256)
    x = Conv2D(int(512*m), k, strides=2, padding="same")(x);
    x = LeakyReLU(alpha=0.2)(x);
    ## (None, 2, 2, 512)
    x = Flatten()(x);
    x = Dropout(0.5)(x);
    outputs = Dense(1)(x);

    return Model(inputs, outputs);

gen_model = GeneratorModel(m=1, k=10);
dis_model = DisclaimerModel(m=1, k=5);

run_test = False;

if(run_test):
    random_seed = tf.random.uniform(shape=(1, rand));
    predicted_image = gen_model(random_seed, training=False);

    print(f"""
    shape: {np.array(predicted_image).shape}
    max: {np.array(predicted_image).max()}
    max(image*22): {np.array(predicted_image).max()}
    max(image*255): {np.array(predicted_image*255).max()}
    """);

    predicted_image_unormalizerd = predicted_image*255;

    print(dis_model(predicted_image_unormalizerd, training=False));

    plt.imshow(tf.squeeze(tf.cast(predicted_image_unormalizerd, tf.int32)));
    plt.show();

binary_crossentropy = losses.BinaryCrossentropy(from_logits=True);

@tf.function
def GenLoss(fake_output):
    loss = binary_crossentropy(tf.ones_like(fake_output), fake_output);
    return loss;

@tf.function
def DisLoss(real_output, fake_output):
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output);
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output);
    loss = real_loss + fake_loss;
    return loss;

@tf.function
def TrainStep(images, batch_size=16, gen_model=gen_model, dis_model=dis_model, gen_optimizer=gen_optimizer, dis_optimizer=dis_optimizer):
    seed = tf.random.uniform(shape=(batch_size, rand));
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        images_pred = gen_model(seed, training=True);

        real_output = dis_model(images, training=True);
        fake_output = dis_model(images_pred*255, training=True);

        gen_loss = GenLoss(fake_output);
        dis_loss = DisLoss(real_output, fake_output);
    
    gen_gradient = gen_tape.gradient(gen_loss, gen_model.trainable_variables);
    dis_gradient = dis_tape.gradient(dis_loss, dis_model.trainable_variables);

    gen_optimizer.apply_gradients(zip(gen_gradient, gen_model.trainable_variables));
    dis_optimizer.apply_gradients(zip(dis_gradient, dis_model.trainable_variables));

    return gen_loss, dis_loss;

def Predict(show=False, save_path=None, seed=tf.random.uniform(shape=(1, rand))):
    image_pred = gen_model(seed, training=False);

    image_pred_unormalizerd = image_pred*255;

    if(show):
        plt.imshow(tf.cast(tf.squeeze(image_pred_unormalizerd), dtype=tf.int32));
        plt.axis(False);
        plt.show();
    
    if(save_path is not None):
        image = Image.fromarray(np.array(np.squeeze(image_pred_unormalizerd), dtype=np.uint8));
        image.save(f"{save_path}.png");

    return image_pred;

save_ckpt = True;

if(save_ckpt):
    checkpoint_path = "Projects/Face generation/Models/DCGAN_checkpoint/train";
    checkpoint_path_2 =  "Projects/Face generation/Models/DCGAN_checkpoint/train_2";

    ckpt = tf.train.Checkpoint(
        gen_model=gen_model,
        dis_model=dis_model,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer);

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5);

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint);
        print ('Latest checkpoint restored!!');

last_epoch = 3055;
epochs = 5000;

for epoch in range(last_epoch, epochs):
    start_time = time.time();
    
    i = 0;

    for image_batch, _ in tqdm.tqdm(data):
        gen_loss, dis_loss = TrainStep(
            images=image_batch,
            batch_size=batch_size);
        if(i % 85 == 0):
            pred_image = Predict(save_path=f"Projects/Face generation/Saves/3/{epoch+1}_{i}");
        i+=1;

    if(save_ckpt):
        if ((epoch + 1) % 5 == 0):
            ckpt_save_path = ckpt_manager.save()
            print (f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}");

    print(f"""
    epoch: {epoch+1}/{epochs} | gen_loss: {gen_loss} | dis_loss: {dis_loss}
    Time: {round(time.time() - start_time, 2)}s
    """);

    pred_image = Predict(save_path=f"Projects/Face generation/Saves/4/{epoch+1}");

pred_image = Predict(show=True);

gen_model.save("Projects/Face generation/Models/DCGAN_200000_1/gen");
dis_model.save("Projects/Face generation/Models/DCGAN_200000_1/dis");
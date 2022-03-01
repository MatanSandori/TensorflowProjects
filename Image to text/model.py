import sys
sys.path.append(".");

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras import utils, layers, applications, Model, activations, metrics, losses, optimizers
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, GlobalAveragePooling2D, ConvLSTM2D, LSTM, GRU, AdditiveAttention, Attention, Reshape, Embedding
from tensorflow.keras.layers.experimental.preprocessing import Resizing, TextVectorization, StringLookup
from tensorflow_helper import helper

import collections
import pathlib
import random
import tqdm
import json
import time
import os

from dataset import dataset

batch_size = 32;
units = 512;

test = {"attention_test": False,
        "encoder_test": False,
        "decoder_test": False,
        "loss_test": False};

data = dataset(load=1000);
ds, tokanize = data.load_all(size=(128, 128), batch_size=batch_size, use_tqdm=True, max_tokens=50000);
vocab_size = tokanize.vectorization.vocabulary_size();

print("\nNumber of tokens: ", vocab_size, "\n");

class B_Attention(layers.Layer):
    def __init__(self, units=units):
        super(B_Attention, self).__init__();

        self.units = units;

        self.dense_0 = Dense(units);
        self.dense_1 = Dense(units);
        self.v = Dense(1);
    def call(self, inputs, hidden, training=False):
        hidden_extra_dim = tf.expand_dims(hidden, axis=1);

        hidden = activations.tanh(
            self.dense_0(inputs, training=training) + self.dense_1(hidden_extra_dim, training=training)
        );

        hidden = self.v(hidden, training=training);

        hidden = activations.softmax(hidden, axis=1);

        x = inputs * hidden;
        x = tf.reduce_sum(x, axis=1);

        return x, hidden;
    def reset_hidden(self, batch_size=batch_size):
        return tf.zeros([batch_size, self.units]);

class Encoder(Model):
    def __init__(self, m=2):
        super(Encoder, self).__init__();

        ## (128, 128, 3)
        ## (64, units)

        self.conv_0 = Conv2D(
            filters=128/m, 
            kernel_size=(1, 1), 
            strides=(2, 2), 
            padding="same",
            activation=activations.relu
        );
        self.conv_1 = Conv2D(
            filters=256/m, 
            kernel_size=(1, 1), 
            strides=(2, 2), 
            padding="same",
            activation=activations.relu
        );
        self.conv_2 = Conv2D(
            filters=512/m, 
            kernel_size=(1, 1), 
            strides=(2, 2), 
            padding="same",
            activation=activations.relu
        );
        self.conv_3 = Conv2D(
            filters=units, 
            kernel_size=(1, 1), 
            strides=(2, 2), 
            padding="same",
            activation=activations.relu
        );
        self.reshape_0 = Reshape(
            target_shape=(64, units)
        );
    def call(self, inputs, training=False):
        x = self.conv_0(
            inputs=inputs, 
            training=training);

        x = self.conv_1(x, training=training);
        x = self.conv_2(x, training=training);
        x = self.conv_3(x, training=training);

        encoder_outputs = self.reshape_0(x);

        return encoder_outputs;

class AttentionInputs(Model):
    def __init__(self, units=units):
        super(AttentionInputs, self).__init__();

        self.outputs = Dense(units);
    def __call__(self, inputs, training=False):
        return self.outputs(inputs, training=training);

class Decoder(Model):
    def __init__(self, units):
        super(Decoder, self).__init__();

        self.units = units;

        self.dense_inputs_processor = Dense(
            units=units, activation=activations.relu
        );

        self.attention = B_Attention(units);

        self.embedding = Embedding(
            input_dim=vocab_size, output_dim=units
        );

        self.gru = GRU(
            units=units,
            return_state=True,
            return_sequences=True,
            activation=activations.tanh);
        
        self.dense = Dense(
            units=units, activation=activations.relu
        );

        self.outputs = self.dense_1 = Dense(
            units=vocab_size)
    def call(self, inputs, attention_inputs, state=None, hidden=None, training=False, batch_size=None):
        if(batch_size == None):
            batch_size = inputs.shape[0];

        if(hidden == None):
            hidden = self.attention.reset_hidden(
                batch_size=batch_size);

        y, hidden = self.attention(attention_inputs, hidden, training=training);
        
        x = self.embedding(inputs, training=training);

        y = tf.expand_dims(y, axis=1);

        x = tf.concat([x, y], axis=-1);

        if(state == None):
            state = self.gru.get_initial_state(x);
        
        x, state = self.gru(x, initial_state=state, training=training);

        x = self.dense(x, training=training);
        
        x = tf.reshape(
            tensor=x, 
            shape=(x.shape[0] * x.shape[1], units)
        );

        decocer_outputs = self.outputs(x, training=training);

        return decocer_outputs, state, hidden;
    def reset_hidden(self, batch_size=batch_size):
        return self.attention.reset_hidden(batch_size);

if(test["attention_test"]):
    attention = B_Attention();

    inputs = tf.random.uniform(shape=(4, units, 348));
    hidden = attention.reset_hidden(4);

    x, hidden = attention(inputs, hidden);
    hidden = tf.squeeze(hidden);

    x, hidden = attention(inputs, hidden);
    hidden = tf.squeeze(hidden);

    print(x, "\n", hidden.shape);

if(test["encoder_test"]):
    encoder = Encoder();

    inputs = tf.random.uniform(shape=(4, 128, 128, 3));

    x = encoder(inputs);

    print(x);

if(test["decoder_test"]):
    inputs = tf.random.uniform(shape=(4, 128, 128, 3));

    encoder = Encoder();
    encoder_outputs = encoder(inputs);
    
    attention_inputs = AttentionInputs(units);
    attention_input = attention_inputs(encoder_outputs);

    dec_input = tf.expand_dims([tokanize.word_to_index("startsentence")] * 4, axis=1);

    decoder = Decoder(units);

    decoder_outputs, state, hidden = decoder(dec_input, attention_input);

    decoder_outputs, state, hidden = decoder(dec_input, attention_input, hidden=hidden);
    hidden = tf.squeeze(hidden);

    print(decoder_outputs);
    print(state.shape, hidden.shape);

encoder = Encoder();
attention_inputs = AttentionInputs(units);
decoder = Decoder(units);

optimizer = optimizers.RMSprop();
loss_obj = losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction=losses.Reduction.NONE
);

def loss(y_true, y_pred):
    _loss = loss_obj(y_true, y_pred);

    mask = tf.not_equal(y_true, 0);
    mask = tf.cast(mask, dtype=_loss.dtype);
    
    _loss *= mask;
    return tf.reduce_mean(_loss);

if(test["loss_test"]):
    rand = tf.random.uniform(shape=(3, 4));

    print(rand);

    print(loss(
        tf.constant([1, 0, 3], dtype=tf.float32),
        rand
    ));

def train_step(images, tokens):
    with tf.GradientTape() as tape:
        hidden = None;
        state = None;

        inputs = encoder(images, training=True);

        attention_input = attention_inputs(inputs, training=True);
        dec_inputs = tf.expand_dims([tokanize.word_to_index("startsentence")] * batch_size, axis=1);

        _loss = tf.constant(value=0, dtype=tf.float32);

        for i in range(1, tokens.shape[1]):
            predictions, state, hidden = decoder(
                inputs=dec_inputs,
                attention_inputs=attention_input,
                state=None,
                hidden=None, 
                training=True);
            
            cur = tokens[:, i];
                        
            _loss += loss(cur, predictions);
            dec_inputs = tf.expand_dims(cur, axis=1);
        
        _loss /= tokens.shape[1];

    trainable_variables = encoder.trainable_variables + attention_inputs.trainable_variables + decoder.trainable_variables;

    gradients = tape.gradient(_loss, trainable_variables);

    optimizer.apply_gradients(zip(gradients, trainable_variables));

    return _loss;

def train(epochs=5):
    def get_avg(total_loss):
        _loss = sum(total_loss) / len(total_loss);
        return _loss.numpy();

    history = {"loss": []};

    for epoch in range(epochs):
        total_loss = [];

        bar = tqdm.tqdm(
            iterable=ds,
            desc="Training",
            total=len(ds)
        );

        for (images, tokens) in bar:
            _loss = train_step(images, tokens);

            total_loss.append(_loss);
            bar.set_description(
                str(get_avg(total_loss)));
        
        _loss = get_avg(total_loss);

        history["loss"].append(_loss);

        print(f"""
        epoch: {epoch + 1}/{epochs} | loss: {_loss}
        """);

    return history;

def predict(image, max_length=50, batch_size=1):
    hidden = None;

    encoder_image = encoder(image, training=False);

    attention_input = attention_inputs(encoder_image, training=False);

    dec_input = tf.expand_dims([tokanize.word_to_index("startsentence") * batch_size], 1);

    end_token = tokanize.word_to_index("endsentence");

    tokens = [];

    for i in range(max_length):
        if(i != 0):
            tokens.append(dec_input.numpy().tolist());

        predictions, hidden, state = decoder(
            inputs=dec_input,
            attention_inputs=attention_input,
            state=None,
            hidden=hidden,
            training=False
        );

        dec_input = tf.random.categorical(predictions, num_samples=1);

        if(dec_input[0][0] == end_token):
            break;

    return tokens;

def tokens_to_string(tokens):
    string = tf.constant("", dtype=tf.string);
    tokens = tf.squeeze(tokens);

    for token in tokens:
        string += tokanize.index_to_word(token);
        string += " ";
    
    return string.numpy();

def get_sampels(num_of_sampels=64):
    images = [];
    _tokens = [];

    for (i, (image, tokens)) in enumerate(ds.unbatch()):
        images.append(image);
        _tokens.append(tokens);

        if(i >= num_of_sampels):
            break;
    
    return images, _tokens;

def plot_history(history):
    pd.DataFrame(history).plot();
    plt.show();

def show_models():
    encoder.summary();
    decoder.summary();

history = train(epochs=80);

show_models();

plot_history(history);

images, _tokens = get_sampels(200);

for (image, _token) in zip(images, _tokens):
    try:
        tokens = predict(tf.expand_dims(image, axis=0));

        print(tf.squeeze(tokens));
        print(tokens_to_string(tokens));

        plt.imshow(image);
        plt.title(tokens_to_string(tokens));
        plt.show();

        print("\n", "-" * 35, "\n");
    except:
        pass;
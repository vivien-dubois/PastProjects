"""
Contains code to generate models
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import stats
import pickle

class NoiseLayer(tf.keras.layers.Layer):
    def __init__(self,mean):
        super(NoiseLayer, self).__init__()
        self.mean = mean

    def get_config(self):
        return {"mean": self.mean}

    def call(self, input):
        return input + tf.random.uniform(tf.shape(input), minval=-0.5 * self.mean, maxval=0.5*self.mean)

def make_generator(noise_magnitude=0):
    image_in = tf.keras.Input(shape=(384,512,3), dtype=tf.float32)
    s1 = tf.keras.layers.Conv2D(32, (9,9), strides=(4,4), padding='same', activation='relu', dtype=tf.float32, name="s1")(image_in)
    s2 = tf.keras.layers.Conv2D(32, (5,5), strides=(2,2), padding='same', activation='relu', dtype=tf.float32, name="s2")(s1)
    s_enc = tf.keras.layers.Conv2D(32, (5,5), strides=(2,2), padding='same', activation='relu', dtype=tf.float32, name="s_enc")(s2)

    if noise_magnitude > 0:
        s_enc = NoiseLayer(noise_magnitude)(s_enc)

    l_dec = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', activation='relu', dtype=tf.float32, name="s_dec")
    l3 = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', activation='relu', dtype=tf.float32, name="s3")
    l_out = tf.keras.layers.Conv2DTranspose(3, (9, 9), strides=(4, 4), padding='same', activation='relu', dtype=tf.float32, name="s_out")
    s_dec = l_dec(s_enc)
    s3 = l3(s_dec)
    s_out = l_out(s3)

    if noise_magnitude > 0:
        enc_in = tf.keras.Input(shape=(24,32,32), dtype=tf.float32)
        s_dec_e = l_dec(enc_in)
        s3_e = l3(s_dec_e)
        s_out_e = l_out(s3_e)
    if noise_magnitude <= 0:
        return tf.keras.Model(inputs=image_in, outputs=s_out)
    else:
        return (tf.keras.Model(inputs=image_in, outputs=s_out), tf.keras.Model(inputs=image_in, outputs=s_enc), tf.keras.Model(inputs=enc_in, outputs=s_out_e))

def get_wadiqam_model_ms(is_diqam=False, loss_fn=tf.keras.losses.MeanAbsoluteError(), lr=0.000001, score_scale=9):
    im_input_trick_16 = tf.keras.Input(shape=(None, 16, 16 ,3), name="x16")
    ref_input_trick_16 = tf.keras.Input(shape=(None, 16, 16 ,3), name="y16")
    im_input_trick_32 = tf.keras.Input(shape=(None, 32, 32 ,3), name="x32")
    ref_input_trick_32 = tf.keras.Input(shape=(None, 32, 32 ,3), name="y32")

    conv11_16=tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_11_16")
    conv12_16=tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_12_16")
    conv21_16=tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_21_16")
    conv22_16=tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_22_16")
    conv31_16=tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_31_16")
    conv32_16=tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_32_16")
    conv41_16=tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_41_16")
    conv42_16=tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_42_16")
    def extractor_trick_16(image):
        conv11 = (conv11_16(image))
        conv12 = (conv12_16(conv11))
        conv21 = (conv21_16(conv12))
        conv22 = (conv22_16(conv21))
        conv31 = (conv31_16(conv22))
        conv32 = (conv32_16(conv31))
        conv41 = (conv41_16(conv32))
        conv42 = (conv42_16(conv41))
        symbolic_shape = K.shape(im_input_trick_16)
        return tf.keras.layers.Lambda(lambda x: tf.reshape(x, (symbolic_shape[0], symbolic_shape[1], 256)))(conv42)
    
    conv11_32=tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_11_32")
    conv12_32=tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_12_32")
    conv21_32=tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_21_32")
    conv22_32=tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_22_32")
    conv31_32=tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_31_32")
    conv32_32=tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_32_32")
    conv41_32=tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_41_32")
    conv42_32=tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_42_32")
    conv51_32=tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_51_32")
    conv52_32=tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_52_32")
    def extractor_trick_32(image):
        conv11 = (conv11_32(image))
        conv12 = (conv12_32(conv11))
        conv21 = (conv21_32(conv12))
        conv22 = (conv22_32(conv21))
        conv31 = (conv31_32(conv22))
        conv32 = (conv32_32(conv31))
        conv41 = (conv41_32(conv32))
        conv42 = (conv42_32(conv41))
        conv51 = (conv51_32(conv42))
        conv52 = (conv52_32(conv51))
        symbolic_shape = K.shape(im_input_trick_32)
        return tf.keras.layers.Lambda(lambda x: tf.reshape(x, (symbolic_shape[0], symbolic_shape[1], 512)))(conv52)

    patches_gen_16 = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 16, 16, 3)))(im_input_trick_16)
    patches_ref_16 = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 16, 16, 3)))(ref_input_trick_16)
    im_c_16 = extractor_trick_16(patches_gen_16)
    ref_c_16 = extractor_trick_16(patches_ref_16)
    patches_gen_32 = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 32, 32, 3)))(im_input_trick_32)
    patches_ref_32 = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 32, 32, 3)))(ref_input_trick_32)
    im_c_32 = extractor_trick_32(patches_gen_32)
    ref_c_32 = extractor_trick_32(patches_ref_32)
    
    im_c = tf.keras.layers.Concatenate(axis=2)([im_c_16, im_c_32])
    ref_c = tf.keras.layers.Concatenate(axis=2)([ref_c_16, ref_c_32])

    subbed = tf.keras.layers.Subtract()([im_c, ref_c])
    dense1 = tf.keras.layers.Dropout(0.55)(tf.keras.layers.Dense(400, activation='relu')(subbed))
    dense2 = tf.keras.layers.Dropout(0.55)(tf.keras.layers.Dense(400, activation='relu')(dense1))
    out_score = tf.keras.layers.Dense(1)(dense2)
    if is_diqam:
        out_score = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1))(out_score)
        out_score = tf.keras.layers.Lambda(lambda x: tf.math.multiply(np.float32(score_scale), tf.math.sigmoid(x)))(out_score)
    else:
        dense1_weights = tf.keras.layers.Dropout(0.55)(tf.keras.layers.Dense(400, activation='relu')(subbed))
        dense2_weights = tf.keras.layers.Dropout(0.55)(tf.keras.layers.Dense(400, activation='relu')(dense1_weights))
        out_weights = tf.keras.layers.Dense(1, activation="softplus", name="out_weights")(dense2_weights)
        weight_sum = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1))(out_weights)
        weight_sum = tf.keras.layers.Lambda(lambda x: tf.math.reciprocal(x))(weight_sum)
        
        out_score = tf.keras.layers.Multiply()([out_score, out_weights])
        out_score = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1))(out_score)
        out_score = tf.keras.layers.Multiply()([out_score, weight_sum])
        out_score = tf.keras.layers.Lambda(lambda x: tf.math.multiply(np.float32(score_scale), tf.math.sigmoid(x)))(out_score)
        

    model = tf.keras.Model(inputs=[im_input_trick_16, ref_input_trick_16, im_input_trick_32, ref_input_trick_32], outputs=out_score)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_fn)
    return model

def get_wadiqam_model_16(is_diqam=False, loss_fn=tf.keras.losses.MeanAbsoluteError(), lr=0.000001, score_scale=9):
    im_input_trick = tf.keras.Input(shape=(None, 16, 16 ,3), name="x")
    ref_input_trick = tf.keras.Input(shape=(None, 16, 16 ,3), name="y")

    i_conv11 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_11")
    i_conv12 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_12")
    i_conv21 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_21")
    i_conv22 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_22")
    i_conv31 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_31")
    i_conv32 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_32")
    i_conv41 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_41")
    i_conv42 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_42")
    def extractor_trick(image):
        conv11 = (i_conv11(image))
        conv12 = (i_conv12(conv11))
        conv21 = (i_conv21(conv12))
        conv22 = (i_conv22(conv21))
        conv31 = (i_conv31(conv22))
        conv32 = (i_conv32(conv31))
        conv41 = (i_conv41(conv32))
        conv42 = (i_conv42(conv41))
        symbolic_shape = K.shape(im_input_trick)
        return tf.keras.layers.Lambda(lambda x: tf.reshape(x, (symbolic_shape[0], symbolic_shape[1], 256)))(conv42)

    patches_gen = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 16, 16, 3)))(im_input_trick)
    patches_ref = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 16, 16, 3)))(ref_input_trick)
    im_c = extractor_trick(patches_gen)
    ref_c = extractor_trick(patches_ref)

    subbed = tf.keras.layers.Subtract()([im_c, ref_c])
    dense1 = tf.keras.layers.Dropout(0.5)(tf.keras.layers.Dense(512, activation='relu')(subbed))
    #dense2 = tf.keras.layers.Dropout(0.5)(tf.keras.layers.Dense(512, activation='relu')(dense1))
    out_score = tf.keras.layers.Dense(1)(dense1)
    if is_diqam:
        out_score = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1))(out_score)
        out_score = tf.keras.layers.Lambda(lambda x: tf.math.multiply(np.float32(score_scale), tf.math.sigmoid(x)))(out_score)
    else:
        dense1_weights = tf.keras.layers.Dropout(0.5)(tf.keras.layers.Dense(512, activation='relu')(subbed))
        #dense2_weights = tf.keras.layers.Dropout(0.5)(tf.keras.layers.Dense(512, activation='relu')(dense1_weights))
        out_weights = tf.keras.layers.Dense(1, activation="softplus", name="out_weights")(dense1_weights)
        weight_sum = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1))(out_weights)
        weight_sum = tf.keras.layers.Lambda(lambda x: tf.math.reciprocal(x))(weight_sum)
        
        out_score = tf.keras.layers.Multiply()([out_score, out_weights])
        out_score = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1))(out_score)
        out_score = tf.keras.layers.Multiply()([out_score, weight_sum])
        out_score = tf.keras.layers.Lambda(lambda x: tf.math.multiply(np.float32(score_scale), tf.math.sigmoid(x)))(out_score)
        

    model = tf.keras.Model(inputs=[im_input_trick, ref_input_trick], outputs=out_score)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_fn)
    return model

def get_wadiqam_model(is_diqam=False, loss_fn=tf.keras.losses.MeanAbsoluteError(), lr=0.000001, score_scale=9, drop_rate=0.5,
                     conv_dropout=False, use_sigmoid=True):
    im_input_trick = tf.keras.Input(shape=(None, 32, 32 ,3), name="x")
    ref_input_trick = tf.keras.Input(shape=(None, 32, 32 ,3), name="y")

    i_conv11 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_11")
    i_conv12 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_12")
    i_conv21 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_21")
    i_conv22 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_22")
    i_conv31 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_31")
    i_conv32 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_32")
    i_conv41 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_41")
    i_conv42 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_42")
    i_conv51 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu', name="conv_51")
    i_conv52 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(2,2), padding='same', activation='relu', name="conv_52")
    def extractor_trick(image):
        if conv_dropout:
            conv11 = tf.keras.layers.Dropout(0.05)(i_conv11(image))
            conv12 = tf.keras.layers.Dropout(0.05)(i_conv12(conv11))
            conv21 = tf.keras.layers.Dropout(0.05)(i_conv21(conv12))
            conv22 = tf.keras.layers.Dropout(0.05)(i_conv22(conv21))
            conv31 = tf.keras.layers.Dropout(0.05)(i_conv31(conv22))
            conv32 = tf.keras.layers.Dropout(0.05)(i_conv32(conv31))
            conv41 = tf.keras.layers.Dropout(0.05)(i_conv41(conv32))
            conv42 = tf.keras.layers.Dropout(0.05)(i_conv42(conv41))
            conv51 = tf.keras.layers.Dropout(0.05)(i_conv51(conv42))
            conv52 = tf.keras.layers.Dropout(0.05)(i_conv52(conv51))
        else:
            conv11 = (i_conv11(image))
            conv12 = (i_conv12(conv11))
            conv21 = (i_conv21(conv12))
            conv22 = (i_conv22(conv21))
            conv31 = (i_conv31(conv22))
            conv32 = (i_conv32(conv31))
            conv41 = (i_conv41(conv32))
            conv42 = (i_conv42(conv41))
            conv51 = (i_conv51(conv42))
            conv52 = (i_conv52(conv51))
        symbolic_shape = K.shape(im_input_trick)
        return tf.keras.layers.Lambda(lambda x: tf.reshape(x, (symbolic_shape[0], symbolic_shape[1], 512)))(conv52)

    patches_gen = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 32, 32, 3)))(im_input_trick)
    patches_ref = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 32, 32, 3)))(ref_input_trick)
    im_c = extractor_trick(patches_gen)
    ref_c = extractor_trick(patches_ref)

    subbed = tf.keras.layers.Subtract()([im_c, ref_c])
    dense1 = tf.keras.layers.Dropout(drop_rate)(tf.keras.layers.Dense(512, activation='relu')(subbed))
    #dense2 = tf.keras.layers.Dropout(0.5)(tf.keras.layers.Dense(512, activation='relu')(dense1))
    out_score = tf.keras.layers.Dense(1)(dense1)
    if is_diqam:
        out_score = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1))(out_score)
        out_score = tf.keras.layers.Lambda(lambda x: tf.math.multiply(np.float32(score_scale), tf.math.sigmoid(x)))(out_score)
    else:
        dense1_weights = tf.keras.layers.Dropout(drop_rate)(tf.keras.layers.Dense(512, activation='relu')(subbed))
        #dense2_weights = tf.keras.layers.Dropout(0.5)(tf.keras.layers.Dense(512, activation='relu')(dense1_weights))
        out_weights = tf.keras.layers.Dense(1, activation="softplus", name="out_weights")(dense1_weights)
        weight_sum = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1))(out_weights)
        weight_sum = tf.keras.layers.Lambda(lambda x: tf.math.reciprocal(x))(weight_sum)
        
        out_score = tf.keras.layers.Multiply()([out_score, out_weights])
        out_score = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1))(out_score)
        out_score = tf.keras.layers.Multiply()([out_score, weight_sum])
        if use_sigmoid:
            out_score = tf.keras.layers.Lambda(lambda x: tf.math.multiply(np.float32(score_scale), tf.math.sigmoid(x)))(out_score)
        

    model = tf.keras.Model(inputs=[im_input_trick, ref_input_trick], outputs=out_score)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_fn)
    return model
import os

import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt
from scipy import stats

import tensorflow as tf

import random
import itertools
import utils

import pickle

def create_fr_image_batch_full_adv(reference_images, mos_scores, shuffle, n_patches=32, n_images=10):
    counter = 0
    while counter < 3000:
        scores = np.zeros((n_images,))
        adversarial_img_dir = "/content/gdrive/My Drive/Colab Notebooks/adversarial_ims2"
        adversarial_names = os.listdir(adversarial_img_dir)
        def key_fun(x):
            return int(x[4:-4])
        adversarial_names = sorted(adversarial_names, key=key_fun)
        x_ims = np.zeros((n_images, 384, 512, 3))
        y_ims = np.zeros((n_images, 384, 512, 3))
        for n_im in range(n_images):
            img_idx = random.choice(np.array(range(25)))
            dist_idx = random.randint(0,119)
            if not shuffle:
                img_idx = int(counter / 120)
                dist_idx = counter % 120
            counter += 1
    
            y_ims[n_im] = reference_images[img_idx]
            i_name = adversarial_img_dir + "/" + adversarial_names[120 * img_idx + dist_idx]
            x_ims[n_im] = plt.imread(i_name).astype(np.float64)[:,:,:3] * 255
            scores[n_im] = mos_scores[120 * img_idx + dist_idx].reshape((1,1))
                
        yield ([np.reshape(x_ims, (n_images, 384,512,3)), np.reshape(y_ims, (n_images, 384,512,3))], scores)

def create_fr_image_batch_full(reference_images, image_split, mos_scores, img_names, shuffle, adv_ims="", decode=True, n_patches=32, patch_size=32, n_images=10, start_idx=0):
    counter = start_idx
    while True:
        scores = np.zeros((n_images,))
        distorted_img_dir = "/content/gdrive/My Drive/tid2013/distorted_images"
        adversarial_img_dir = "/content/gdrive/My Drive/Colab Notebooks/adversarial_ims"
        adversarial_names = []
        max_rand = 119
        if adv_ims != "":
            max_rand = 270
            adversarial_img_dir = "/content/gdrive/My Drive/Colab Notebooks/" + adv_ims
            adversarial_names = os.listdir(adversarial_img_dir)
        def key_fun(x):
            return int(x[4:-4])
        adversarial_names = sorted(adversarial_names, key=key_fun)
        x_ims = np.zeros((n_images, 384, 512, 3))
        y_ims = np.zeros((n_images, 384, 512, 3))
        for n_im in range(n_images):
            img_idx = random.choice(image_split)
            dist_idx = random.randint(0,270)
            if not shuffle:
                img_idx = image_split[int(counter / 120)]
                dist_idx = counter % 120
            counter += 1
    
            y_ims[n_im] = reference_images[img_idx]
            if dist_idx < 120:
                if decode:
                    i_name = (img_names[120 * img_idx + dist_idx]).decode('utf-8')
                else:
                    i_name = (img_names[120 * img_idx + dist_idx])
                x_ims[n_im] = plt.imread(distorted_img_dir + "/" + i_name).astype(np.float64) + np.random.normal(scale=0.05, size=y_ims[n_im].shape)# / 255 - 0.5
                scores[n_im] = mos_scores[120 * img_idx + dist_idx].reshape((1,1))
            elif dist_idx < 240:
                i_name = adversarial_img_dir + "/" + adversarial_names[120 * img_idx + dist_idx - 120]
                x_ims[n_im] = plt.imread(i_name).astype(np.float64)[:,:,:3] * 255
                scores[n_im] = mos_scores[120 * img_idx + dist_idx - 120].reshape((1,1))
            elif dist_idx < 265:
                x_ims[n_im] = y_ims[n_im]# + np.random.normal(scale=0.05, size=y_im.shape)
                scores[n_im] = np.float64(9)
                """
                if dist_idx > 195:
                    scores[n_im] = np.float64(0)
                    means = np.mean(x_im, 2)
                    for channel in range(3):
                        x_im[:,:,channel] = means
                """
            else:
                x_ims[n_im] = reference_images[(img_idx + random.randint(1,24)) % 25] + np.random.normal(scale=1, size=y_ims[n_im].shape)
                scores[n_im] = np.float64(0)
                
        yield ([np.reshape(x_ims, (n_images, 384,512,3)), np.reshape(y_ims, (n_images, 384,512,3))], scores)

def create_fr_image_batch_ms(reference_images, image_split, mos_scores, img_names, shuffle, decode=True, n_patches=32, patch_size=32, n_images=10, adv_ims=""):
    counter = 0
    while True:
        x_res = np.zeros((n_images, n_patches, patch_size, patch_size, 3))
        y_res = np.zeros((n_images, n_patches, patch_size, patch_size, 3))
        x_res_ms = np.zeros((n_images, n_patches, patch_size//2, patch_size//2, 3))
        y_res_ms = np.zeros((n_images, n_patches, patch_size//2, patch_size//2, 3))
        scores = np.zeros((n_images,))
        distorted_img_dir = "/content/gdrive/My Drive/tid2013/distorted_images"
        adversarial_img_dir = "/content/gdrive/My Drive/Colab Notebooks/adversarial_ims"
        adversarial_names = []
        max_rand = 119
        if adv_ims != "":
            max_rand = 280
            adversarial_img_dir = "/content/gdrive/My Drive/Colab Notebooks/" + adv_ims
            adversarial_names = os.listdir(adversarial_img_dir)
        for n_im in range(n_images):
            img_idx = random.choice(image_split)
            dist_idx = random.randint(0,max_rand)
            if not shuffle:
                img_idx = image_split[int(counter / 120)]
                dist_idx = counter % 120
            counter += 1
    
            y_im = reference_images[img_idx]
            if dist_idx < 120:
                if decode:
                    i_name = (img_names[120 * img_idx + dist_idx]).decode('utf-8')
                else:
                    i_name = (img_names[120 * img_idx + dist_idx])
                x_im = plt.imread(distorted_img_dir + "/" + i_name).astype(np.float64) + np.random.normal(scale=0.05, size=y_im.shape)# / 255 - 0.5
                scores[n_im] = mos_scores[120 * img_idx + dist_idx].reshape((1,1))
            elif dist_idx < 240:
                i_name = adversarial_img_dir + "/" + adversarial_names[120 * img_idx + dist_idx - 120]
                x_im = plt.imread(i_name).astype(np.float64)[:,:,:3] * 255
                scores[n_im] = mos_scores[120 * img_idx + dist_idx - 120].reshape((1,1))
            elif dist_idx < 275:
                x_im = y_im# + np.random.normal(scale=0.05, size=y_im.shape)
                scores[n_im] = np.float64(10)
                """
                if dist_idx > 195:
                    scores[n_im] = np.float64(0)
                    means = np.mean(x_im, 2)
                    for channel in range(3):
                        x_im[:,:,channel] = means
                """
            else:
                x_im = reference_images[(img_idx + random.randint(1,24)) % 25] + np.random.normal(scale=1, size=y_im.shape)
                scores[n_im] = np.float64(0)
                
            y_im += np.random.normal(scale=0.025, size=y_im.shape)
            for i in range(n_patches):
                ul1 = np.random.randint(x_im.shape[0]-patch_size)
                ul2 = np.random.randint(x_im.shape[1]-patch_size)
                x_res[n_im][i] = x_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
                y_res[n_im][i] = y_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
                ps_4 = int(patch_size / 4)
                x_res_ms[n_im][i] = x_im[ul1+ps_4:ul1+3*ps_4, ul2+ps_4:ul2+3*ps_4]
                y_res_ms[n_im][i] = y_im[ul1+ps_4:ul1+3*ps_4, ul2+ps_4:ul2+3*ps_4]
        yield ([np.reshape(x_res, (n_images, n_patches, patch_size, patch_size, 3)),
                np.reshape(y_res, (n_images, n_patches, patch_size, patch_size, 3)),
                np.reshape(x_res_ms, (n_images, n_patches, patch_size//2, patch_size//2, 3)),
                np.reshape(y_res_ms, (n_images, n_patches, patch_size//2, patch_size//2, 3))],
                scores)

def create_fr_image_batch(reference_images, image_split, mos_scores, img_names, shuffle, decode=True, n_patches=32, patch_size=32, n_images=10, adv_ims=""):
    counter = 0
    while True:
        x_res = np.zeros((n_images, n_patches, patch_size, patch_size, 3))
        y_res = np.zeros((n_images, n_patches, patch_size, patch_size, 3))
        scores = np.zeros((n_images,))
        distorted_img_dir = "/content/gdrive/My Drive/tid2013/distorted_images"
        adversarial_img_dir = "/content/gdrive/My Drive/Colab Notebooks/adversarial_ims"
        adversarial_names = []
        max_rand = 119
        if adv_ims != "":
            max_rand = 280
            adversarial_img_dir = "/content/gdrive/My Drive/Colab Notebooks/" + adv_ims
            adversarial_names = os.listdir(adversarial_img_dir)
        for n_im in range(n_images):
            img_idx = random.choice(image_split)
            dist_idx = random.randint(0,max_rand)
            if not shuffle:
                img_idx = image_split[int(counter / 120)]
                dist_idx = counter % 120
            counter += 1
    
            y_im = reference_images[img_idx]
            if dist_idx < 120:
                if decode:
                    i_name = (img_names[120 * img_idx + dist_idx]).decode('utf-8')
                else:
                    i_name = (img_names[120 * img_idx + dist_idx])
                x_im = plt.imread(distorted_img_dir + "/" + i_name).astype(np.float64) + np.random.normal(scale=0.05, size=y_im.shape)# / 255 - 0.5
                scores[n_im] = mos_scores[120 * img_idx + dist_idx].reshape((1,1))
            elif dist_idx < 240:
                i_name = adversarial_img_dir + "/" + adversarial_names[120 * img_idx + dist_idx - 120]
                x_im = plt.imread(i_name).astype(np.float64)[:,:,:3] * 255
                scores[n_im] = mos_scores[120 * img_idx + dist_idx - 120].reshape((1,1))
            elif dist_idx < 275:
                x_im = y_im# + np.random.normal(scale=0.05, size=y_im.shape)
                scores[n_im] = np.float64(10)
                """
                if dist_idx > 195:
                    scores[n_im] = np.float64(0)
                    means = np.mean(x_im, 2)
                    for channel in range(3):
                        x_im[:,:,channel] = means
                """
            else:
                x_im = reference_images[(img_idx + random.randint(1,24)) % 25] + np.random.normal(scale=1, size=y_im.shape)
                scores[n_im] = np.float64(0)
                
            y_im += np.random.normal(scale=0.025, size=y_im.shape)
            for i in range(n_patches):
                ul1 = np.random.randint(x_im.shape[0]-patch_size)
                ul2 = np.random.randint(x_im.shape[1]-patch_size)
                x_res[n_im][i] = x_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
                y_res[n_im][i] = y_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
        yield ([np.reshape(x_res, (n_images, n_patches, patch_size, patch_size, 3)),
                np.reshape(y_res, (n_images, n_patches, patch_size, patch_size, 3))],
                scores)

def aa_model_fn_metric(learning_rate=0.00001, loss_fn=tf.math.reduce_sum,
                       get_metric_fn=utils.get_model_fn):

    metric_model = tf.estimator.Estimator(model_fn=get_metric_fn(learning_rate, loss_fn=loss_fn),
                                              model_dir="tensorflow_baseline_TID_OG")
    
    def model_fn(features, labels, mode):
        net = tf.layers.conv2d(features, 32, [9, 9], strides=4, padding='SAME')
        print(net)
        net = tf.layers.conv2d(net, 32, [5, 5], strides=2, padding='SAME')
        net = tf.layers.conv2d(net, 32, [5, 5], strides=2, padding='SAME')
        print(net)
        net = tf.layers.conv2d_transpose(net, 32, [5, 5], strides=2, padding='SAME')
        net = tf.layers.conv2d_transpose(net, 32, [5, 5], strides=2, padding='SAME')
        net = tf.layers.conv2d_transpose(net, 3, [9, 9], strides=4, padding='SAME')
        print(net)

        predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
            "output_image": net
        }
        
        print(features)
        def fnfn():
            net_p = tf.image.extract_image_patches(net, [1,32,32,1], [1,32,32,1], [1,1,1,1], 'SAME')
            net_p = tf.reshape(net_p, [1, 192, 32, 32, 3])
            feat_p = tf.image.extract_image_patches(features, [1,32,32,1], [1,32,32,1], [1,1,1,1], 'SAME')
            feat_p = tf.reshape(feat_p, [1, 192, 32, 32, 3])
            return ({"x": net_p, "y": feat_p})
        for el in metric_model.predict(input_fn=fnfn, yield_single_examples=False):
            scores = el

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions["output_image"])

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = loss_fn(scores)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-04)
            train_op = optimizer.maximize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {}
        for k in eval_metrics:
            eval_metrics[k] = eval_metrics[k](labels=features, predictions=predictions["output_image"])
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    return model_fn

def aa_generator(dir_name="tid2013/reference_images"):
    names = os.listdir(dir_name)
    for name in names:
        if ".bmp" not in name and ".BMP" not in name:
            continue
            
        yield (plt.imread(dir_name + "/" + name).astype(np.float32), np.array(np.float32(0)).reshape((1)))

def aa_model_fn(learning_rate=0.00001, loss_fn=tf.losses.mean_squared_error):

    def model_fn(features, labels, mode):
        net = tf.layers.conv2d(features, 32, [9, 9], strides=4, padding='SAME')
        net = tf.layers.conv2d(net, 32, [5, 5], strides=2, padding='SAME')
        net = tf.layers.conv2d(net, 32, [5, 5], strides=2, padding='SAME')
        net = tf.layers.conv2d_transpose(net, 32, [5, 5], strides=2, padding='SAME')
        net = tf.layers.conv2d_transpose(net, 32, [5, 5], strides=2, padding='SAME')
        net = tf.layers.conv2d_transpose(net, 3, [9, 9], strides=4, padding='SAME')

        predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
            "output_image": net
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions["output_image"])

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = loss_fn(labels=features, predictions=predictions["output_image"])

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-04)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {}
        for k in eval_metrics:
            eval_metrics[k] = eval_metrics[k](labels=features, predictions=predictions["output_image"])
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    return model_fn
import os

import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt
from scipy import stats

import tensorflow as tf

import random
import itertools

import pickle

def read_images(dir_name):
    result = []
    image_names = sorted(os.listdir(dir_name))
    print(image_names)
    for img_name in image_names:
        result.append(plt.imread(dir_name + "/" + img_name))
    return np.array(result)

def read_scores(file_name):
    with open(file_name, "r") as file:
        result = []
        for line in file.readlines():
            result.append(float(line))
        return np.array(result)
        
def create_fr_image_batch(reference_images, image_split, mos_scores, img_names, shuffle, n_patches=32, patch_size=32):
    counter = 0
    while counter < image_split.shape[0] * 120:
        x_res = np.zeros((n_patches, patch_size, patch_size, 3))
        y_res = np.zeros((n_patches, patch_size, patch_size, 3))
        distorted_img_dir = "/content/gdrive/My Drive/tid2013/distorted_images"
        img_idx = random.choice(image_split)
        dist_idx = random.randint(0,119)
        if not shuffle:
            img_idx = image_split[int(counter / 120)]
            dist_idx = counter % 120
        counter += 1

        y_im = reference_images[img_idx]
        if dist_idx < 120:
            x_im = plt.imread(distorted_img_dir + "/" + (img_names[120 * img_idx + dist_idx]).decode('utf-8')).astype(np.float32)# / 255 - 0.5
            score = mos_scores[120 * img_idx + dist_idx].reshape((1))
        else:
            x_im = y_im
            score = np.array([9])
            
        for i in range(n_patches):
            ul1 = np.random.randint(x_im.shape[0]-patch_size)
            ul2 = np.random.randint(x_im.shape[1]-patch_size)
            x_res[i] = x_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
            y_res[i] = y_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
        yield ({"x": x_res, "y": y_res}, score)
               
def get_model_fn(learning_rate=0.00001, loss_fn=tf.losses.absolute_difference, weighted_average=True,
                 eval_metrics={"MAE": tf.metrics.mean_absolute_error}):

    def cnn_model_fn(features, labels, mode):
        patches_per_image = 0
        print("\nFeatures shape: " + str(features["x"].shape))
        patches_per_image = features["x"].shape[1]
        
        """Model function for CNN."""
        with tf.device('/cpu:0'):
            input_layer = features["x"]
            ref_input_layer = features["y"]

        def process_image(image):
            with tf.device('/gpu:0'):
                # Convolutional Layer #1
                conv1 = tf.layers.conv2d(inputs=image,filters=32,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                conv12 = tf.layers.conv2d(inputs=conv1,filters=32,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                # Pooling Layer #1
                pool1 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2, 2], strides=2)

                # Convolutional Layer #2
                conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                conv22 = tf.layers.conv2d(inputs=conv2,filters=64,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv22, pool_size=[2, 2], strides=2)

                # Convolutional Layer #3
                conv3 = tf.layers.conv2d(inputs=pool2,filters=128,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                conv32 = tf.layers.conv2d(inputs=conv3,filters=128,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv32, pool_size=[2, 2], strides=2)

                # Convolutional Layer #4
                conv4 = tf.layers.conv2d(inputs=pool3,filters=256,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                conv42 = tf.layers.conv2d(inputs=conv4,filters=256,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                pool4 = tf.layers.max_pooling2d(inputs=conv42, pool_size=[2, 2], strides=2)

                # Convolutional Layer #5
                conv5 = tf.layers.conv2d(inputs=pool4,filters=512,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                conv52 = tf.layers.conv2d(inputs=conv5,filters=512,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                pool5 = tf.layers.max_pooling2d(inputs=conv52, pool_size=[2, 2], strides=2)
                print("\npool5\n" + str(pool5))

                # Dense Layer
                pool5_flat = tf.reshape(pool5, [patches_per_image, 512])
                return pool5_flat

        outputs = tf.map_fn(process_image, input_layer)
        print("\noutputs\n" + str(outputs))
        ref_outputs = tf.map_fn(process_image, ref_input_layer)
        print("\nref_outputs\n" + str(ref_outputs))

        #conced_features = tf.reshape(tf.concat([outputs, ref_outputs], 2), [-1, patches_per_image, ref_outputs.shape[2] * 2])
        conced_features = tf.math.subtract(outputs, ref_outputs)
        print("\nconced\n" + str(conced_features))

        dense1 = tf.layers.dense(inputs=conced_features, units=512, activation=tf.nn.relu)
        drop1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
        print("\ndense1\n" + str(dense1))

        # Output Layer
        output = tf.layers.dense(inputs=drop1, units=1, activation=None)
        print("\noutput\n" + str(output))

        if not weighted_average:
            scores = tf.reduce_mean(output, 1)
            print("\nscores\n" + str(scores))
        else:
            dense_weight1 = tf.layers.dense(inputs=conced_features, units=512, activation=tf.nn.relu)
            drop_weight1 = tf.layers.dropout(inputs=dense_weight1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
            output_weight = tf.math.exp(tf.layers.dense(inputs=drop_weight1, units=1, activation=None))
            weight_sum = tf.math.add(tf.math.reduce_sum(output_weight, 1), 1e-04)
            scores = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(output_weight, output), 1), weight_sum)

        predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
            "score": scores
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions["score"])

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = loss_fn(labels=labels, predictions=predictions["score"])

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-04)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {}
        for k in eval_metrics:
            eval_metrics[k] = eval_metrics[k](labels=labels, predictions=predictions["score"])
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    return cnn_model_fn
    
def get_model_fn_64(learning_rate=0.00001, loss_fn=tf.losses.absolute_difference, weighted_average=True,
                 eval_metrics={"MAE": tf.metrics.mean_absolute_error}):

    def cnn_model_fn(features, labels, mode):
        """Model function for CNN."""
        patches_per_image = features["x"].shape[1]
        with tf.device('/cpu:0'):
            input_layer = features["x"]
            ref_input_layer = features["y"]
        
        def process_image(image):
            with tf.device('/gpu:0'):
                # Convolutional Layer #1
                conv1 = tf.layers.conv2d(inputs=image,filters=64,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                conv12 = tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                # Pooling Layer #1
                pool1 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2, 2], strides=2)

                # Convolutional Layer #2
                conv2 = tf.layers.conv2d(inputs=pool1,filters=128,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                conv22 = tf.layers.conv2d(inputs=conv2,filters=128,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv22, pool_size=[2, 2], strides=2)

                # Convolutional Layer #3
                conv3 = tf.layers.conv2d(inputs=pool2,filters=256,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                conv32 = tf.layers.conv2d(inputs=conv3,filters=256,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv32, pool_size=[2, 2], strides=2)

                # Convolutional Layer #4
                conv4 = tf.layers.conv2d(inputs=pool3,filters=512,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                conv42 = tf.layers.conv2d(inputs=conv4,filters=512,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                pool42 = tf.layers.max_pooling2d(inputs=conv42, pool_size=[2, 2], strides=2)

                # Convolutional Layer #5
                conv5 = tf.layers.conv2d(inputs=pool42,filters=1024,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                conv52 = tf.layers.conv2d(inputs=conv5,filters=1024,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                pool5 = tf.layers.max_pooling2d(inputs=conv52, pool_size=[2, 2], strides=2)
                
                # Convolutional Layer #5
                conv6 = tf.layers.conv2d(inputs=pool5,filters=1024,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                conv62 = tf.layers.conv2d(inputs=conv6,filters=1024,kernel_size=[3, 3],
                    padding="same",activation=tf.nn.relu)
                pool6 = tf.layers.max_pooling2d(inputs=conv62, pool_size=[2, 2], strides=2)

                # Dense Layer
                pool6_flat = tf.reshape(pool6, [patches_per_image, 1024])
                return pool6_flat

        outputs = tf.map_fn(process_image, input_layer)
        print("\noutputs\n" + str(outputs))
        ref_outputs = tf.map_fn(process_image, ref_input_layer)
        print("\nref_outputs\n" + str(ref_outputs))

        conced_features = tf.reshape(tf.concat([outputs, ref_outputs], 2), [-1, patches_per_image, ref_outputs.shape[2] * 2])
        print("\nconced\n" + str(conced_features))

        dense1 = tf.layers.dense(inputs=conced_features, units=1024, activation=tf.nn.relu)
        # Output Layer
        output = tf.layers.dense(inputs=dense1, units=1, activation=None)

        if not weighted_average:
            scores = tf.reduce_mean(output, 1)
        else:
            dense_weight1 = tf.layers.dense(inputs=conced_features, units=512, activation=tf.nn.relu)
            output_weight = tf.math.abs(tf.layers.dense(inputs=dense_weight1, units=1, activation=None))
            weight_sum = tf.math.add(tf.math.reduce_sum(output_weight, 1), 1e-06)
            scores = tf.math.divide_no_nan(tf.math.reduce_sum(tf.math.multiply(output_weight, output), 1), tf.math.reduce_sum(output_weight, 1))

        predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
            "score": scores
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions["score"])

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = loss_fn(labels=labels, predictions=predictions["score"])

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-05)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {}
        for k in eval_metrics:
            eval_metrics[k] = eval_metrics[k](labels=labels, predictions=predictions["score"])
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    return cnn_model_fn
    
def train_on_split(dir_name, ref_images, scores, restore_values=True, learning_rate=0.000005, n_loops=20, steps_per_loop=3000, split=None,
                  loss_fn=tf.losses.mean_squared_error, eval_metrics={}, batch_size=4, eval_fn=None, train_fn=None,
                  n_eval_images=4, eval_scores=None, patch_size=32, n_patches=32, model_fn_gen=get_model_fn, image_names=None):
    if split is None:
        split = np.random.permutation(np.array(range(24)))
        if restore_values:
            try:
                split = pickle.load(open(dir_name + "/split.pkl", "rb"))
            except:
                print("\nCould not find saved split, creating random permutation\n")
        
    basic_classifier = tf.estimator.Estimator(model_fn=model_fn_gen(learning_rate, loss_fn=loss_fn, eval_metrics=eval_metrics),
                                              model_dir=dir_name)
    
    def train_input_fn():
        with (tf.device('/cpu:0')):
            ds = tf.data.Dataset.from_generator(create_fr_image_batch, ({"x": tf.float32, "y": tf.float32}, tf.float32),
                                            ({"x": tf.TensorShape([n_patches,patch_size,patch_size,3]), "y": tf.TensorShape([n_patches,patch_size,patch_size,3])}, tf.TensorShape([1])),
                                           args=(ref_images, split[:len(split)-n_eval_images], scores, image_names, True, n_patches, patch_size))

            ds_batches = ds.repeat().batch(batch_size).prefetch(2)
            return ds_batches

    def eval_input_fn():
        with (tf.device('/cpu:0')):
            ds = tf.data.Dataset.from_generator(create_fr_image_batch, ({"x": tf.float32, "y": tf.float32}, tf.float32),
                                            ({"x": tf.TensorShape([192,patch_size,patch_size,3]), "y": tf.TensorShape([192,patch_size,patch_size,3])}, tf.TensorShape([1])),
                                           args=(ref_images, split[len(split)-n_eval_images:], scores, image_names, False, 192, patch_size))

            ds_batches = ds.batch(batch_size).repeat(1).prefetch(2)
            return ds_batches
        
    if eval_fn is None:
        eval_fn = eval_input_fn
    if train_fn is None:
        train_fn = train_input_fn
        
    evaluation_results_list = []
    correlation_results_list = []
    if restore_values:
        try:
            evaluation_results_list = pickle.load(open(dir_name + "/evaluation_results.pkl", "rb"))
        except:
            print("\nCould not find saved evaluation results, creating empty list\n")
        try:
            correlation_results_list = pickle.load(open(dir_name + "/correlation_results.pkl", "rb"))
        except:
            print("\nCould not find correlation results...\n")
    
    if n_loops > 0:
        for i in range(n_loops):
            basic_classifier.train(
                input_fn=train_fn,
                steps=steps_per_loop)

            """
            eval_results = basic_classifier.evaluate(input_fn=eval_fn)
            evaluation_results_list.append(eval_results["loss"])

            pickle.dump(evaluation_results_list, open(dir_name + "/evaluation_results.pkl", "wb"))
            """
            pickle.dump(split, open(dir_name + "/split.pkl", "wb"))

            predictions_gen = basic_classifier.predict(input_fn=eval_fn)
            predictions = []
            for value in predictions_gen:
                predictions.append(value)
            predictions = np.array(predictions)

            if not eval_scores is None:
                correlation_results_list.append(scipy.stats.spearmanr(predictions.flatten(), eval_scores.flatten()))
                correlation_results_list.append(scipy.stats.pearsonr(predictions.flatten(), eval_scores.flatten()))
                print("Correlation results:")
                print(correlation_results_list[-2:])
                print("End correlation results")
                pickle.dump(correlation_results_list, open(dir_name + "/correlation_results.pkl", "wb"))
    else:
        predictions_gen = basic_classifier.predict(input_fn=eval_fn)
        predictions = []
        for value in predictions_gen:
            predictions.append(value)
        predictions = np.array(predictions)
        
    return (split, evaluation_results_list, basic_classifier, predictions, correlation_results_list)
    
def process_results(results, n_valid=4):
    validScores = scores[np.array([i for i in [range(j*120, (j+1)*120) for j in results[0][len(results[0]) - n_valid:]]]).flatten()]
    print(stats.spearmanr(results[3].flatten(), validScores))
    print(stats.pearsonr(results[3].flatten(), validScores))
    plt.scatter(results[3].flatten(), validScores)
    
def correlation_coefficient(labels, predictions, _lambda=0):
    x = labels
    y = predictions
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm = tf.math.subtract(x, mx)
    ym = tf.math.subtract(y, my)
    r_num = tf.math.reduce_sum(tf.math.multiply(xm,ym))
    r_den = tf.math.sqrt(tf.math.multiply(tf.math.reduce_sum(tf.math.square(xm)), tf.math.reduce_sum(tf.math.square(ym))))
    r = tf.math.divide_no_nan(r_num, r_den)
    
    r = tf.math.maximum(tf.math.minimum(r, 1.0), -1.0)
    if _lambda > 0:
        return tf.math.add(1 - r, tf.math.multiply(_lambda, tf.math.reduce_mean(tf.math.abs(tf.math.subtract(x,y)))))
    else:
        return 1 - tf.math.square(r)

def create_fr_image_batch_LIVE(image_folder, image_split, scores, orgs, n_patches=32, patch_size=32):
    refs = scipy.io.loadmat("databaserelease2/databaserelease2/refnames_all.mat")
    dmos = scipy.io.loadmat("databaserelease2/databaserelease2/dmos.mat")
    n_p = n_patches
    counter = 0
    image_folder = image_folder.decode('utf-8')
    img_files = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]
    
    #Skip counters
    notI = 0
    org = 0
    badS = 0
    
    for file_ext in img_files:
        print("Using", file_ext)
        name_list = os.listdir(image_folder + "/" + file_ext)
        name_list = [name for name in name_list if ".bmp" in name]
        for img_name in sorted(name_list, key=lambda x: int(x[3:-4])):
            if not ".bmp" in img_name:
                notI += 1
                continue
                
            x_im = plt.imread(image_folder + "/" + file_ext + "/" + img_name).astype(np.float32)# / 255 - 0.5
            y_im = plt.imread("databaserelease2/databaserelease2/refimgs/" + refs["refnames_all"][0][counter][0]).astype(np.float32)# / 255 - 0.5
            score = scores[counter]

            if orgs[counter] == 1:
                org += 1
                counter += 1
                continue

            counter += 1

            if x_im.shape != y_im.shape:
                badS += 1
                continue
            
            x_res = np.zeros((n_p,patch_size,patch_size,3))
            y_res = np.zeros((n_p,patch_size,patch_size,3))
            for i in range(n_p):
                ul1 = np.random.randint(x_im.shape[0]-patch_size)
                ul2 = np.random.randint(x_im.shape[1]-patch_size)
                x_res[i] = x_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
                y_res[i] = y_im[ul1:ul1+patch_size, ul2:ul2+patch_size]

            yield ({"x": x_res, "y": y_res}, np.array(9 - (0.09 * score)).reshape((1,)))
    print("Invalid names: " + str(notI))
    print("Originals: " + str(org))
    print("Bad ref shapes: " + str(badS))
          
def get_eval_input_fn_LIVE(n_patches, patch_size):
    def eval_input_fn():
        with (tf.device('/cpu:0')):
            ds = tf.data.Dataset.from_generator(create_fr_image_batch_LIVE, ({"x": tf.float32, "y": tf.float32}, tf.float32),
                                           ({"x": tf.TensorShape([n_patches,patch_size,patch_size,3]), "y": tf.TensorShape([n_patches,patch_size,patch_size,3])}, tf.TensorShape([1])),
                                           args=("databaserelease2/databaserelease2", np.array(range(29)), dmos["dmos"][0].flatten(),
                                                dmos["orgs"][0].flatten()))

            ds_batches = ds.batch(4).repeat(1).prefetch(2)
            return ds_batches
        
    return eval_input_fn
"""
Contains helper functions and other utility code
"""

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
import xlrd
from PIL import Image

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

def correlation_loss(y_true, y_pred, _lambda=0):
    y = ops.convert_to_tensor(y_pred)
    x = math_ops.cast(y_true, y_pred.dtype)
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

def train_on_generators(folder, train_gen, val_gen=None, model_fn=None, model=None, steps_per_epoch=None, epochs=1, save_freq=2,
                        val_steps=None, restore_last_model=True, samples_per_epoch=3000):
    try:
        os.mkdir(folder)
        print("Dir created")
    except FileExistsError:
        print("Dir exists")
        
    if model_fn is not None and model is not None:
        print("Both model_fn and model are set.")
        return None
    
    if model_fn is not None:
        model = model_fn()
    if model is None:
        print("No model found: returning.")
        return None
    
    if restore_last_model:
        try:
            model.load_weights("{0}/model.hdf5".format(folder))
        except:
            print("No model was found in model directory.")
        
    old_hist = None
    try:
        with open("{0}/history.pkl".format(folder), "rb") as file:
            old_hist = pickle.load(file)
    except:
        print("No history found in model directory.")
    
    def cond_save(epoch):
        if epoch % save_freq == 0:
            model.save_weights("{0}/model_{1}.hdf5".format(folder, epoch))
    cb_list = [
        tf.keras.callbacks.ModelCheckpoint(folder + "/best_model.hdf5", save_best_only=True, save_weights_only=True,
                                           save_freq="epoch"),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda e,logs: cond_save(e))
    ]
    new_hist = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=val_gen,
                           validation_steps=val_steps, validation_freq=1, shuffle=False, callbacks=cb_list)
        
    model.save_weights("{0}/model.hdf5".format(folder))
    if old_hist is None:
        with open("{0}/history.pkl".format(folder), "wb") as file:
            pickle.dump(new_hist.history, file)
        return new_hist, new_hist.history
    else:
        old_hist["loss"].extend(new_hist.history["loss"])
        if "val_loss" in old_hist and "val_loss" in new_hist.history:
            old_hist["val_loss"].extend(new_hist.history["val_loss"])
        with open("{0}/history.pkl".format(folder), "wb") as file:
            pickle.dump(old_hist, file)
        return new_hist, old_hist
    
def compute_objective_scores(generator_name, ref_images, metrics=[], n_patches=0, im_dtype=tf.float32):
    gen = tf.keras.models.load_model(generator_name)
    ref_tensor = tf.convert_to_tensor(ref_images, tf.float32)
    images = tf.cast(gen(tf.convert_to_tensor(ref_images, im_dtype)), tf.float32)
    psnrs = tf.image.psnr(images, ref_images.astype(np.float32), max_val=255.0)
    ssims = tf.image.ssim_multiscale(images, ref_images.astype(np.float32), max_val=255.0)
    if n_patches == 0:
        metric_results = [np.mean(m([tf.convert_to_tensor(images, dtype=im_dtype), tf.convert_to_tensor(ref_images, dtype=im_dtype)])) for m in metrics]
    else:
        metric_results = []
        for metric in metrics:
            in_list = []
            in_list_ref = []
            for j in range(n_patches):
                ul1 = np.random.randint(384-32)
                ul2 = np.random.randint(512-32)
                new_in = images[:, ul1:ul1+32, ul2:ul2+32, :]
                in_list.append(new_in)
                new_in_ref = ref_tensor[:, ul1:ul1+32, ul2:ul2+32, :]
                in_list_ref.append(new_in_ref)
            input_tensor = tf.stack(in_list, 1)
            input_tensor_ref = tf.stack(in_list_ref, 1)
            metric_results.append(np.mean(metric([input_tensor, input_tensor_ref])))

    res = [np.mean(ssims), np.mean(psnrs)]
    res.append(generator_loss(images, ref_images).numpy())
    res.extend(metric_results)
    
    return res
    
def compute_objective_scores_folder(folder_name, ref_images, key_fun=None, metrics=[], n_patches=0, im_dtype=tf.float32):
    images = tf.convert_to_tensor(read_images(folder_name, key_fun) * 255)[:,:,:,:3]
    ref_tensor = tf.convert_to_tensor(ref_images, tf.float32)
    psnrs = tf.image.psnr(images, ref_images.astype(np.float32), max_val=255.0)
    ssims = tf.image.ssim_multiscale(images, ref_images.astype(np.float32), max_val=255.0)
    if n_patches == 0:
        metric_results = [np.mean(m([tf.convert_to_tensor(images, dtype=im_dtype), tf.convert_to_tensor(ref_images, dtype=im_dtype)])) for m in metrics]
    else:
        metric_results = []
        for metric in metrics:
            in_list = []
            in_list_ref = []
            for j in range(n_patches):
                ul1 = np.random.randint(384-32)
                ul2 = np.random.randint(512-32)
                new_in = images[:, ul1:ul1+32, ul2:ul2+32, :]
                in_list.append(new_in)
                new_in_ref = ref_tensor[:, ul1:ul1+32, ul2:ul2+32, :]
                in_list_ref.append(new_in_ref)
            input_tensor = tf.stack(in_list, 1)
            input_tensor_ref = tf.stack(in_list_ref, 1)
            metric_results.append(np.mean(metric([input_tensor, input_tensor_ref])))

    res = [np.mean(ssims), np.mean(psnrs)]
    res.append(generator_loss(images, ref_images).numpy())
    res.extend(metric_results)

    return res

def read_images(dir_name, key_fun=None):
    result = []
    image_names = sorted(os.listdir(dir_name), key=key_fun)
    for img_name in image_names:
        result.append(plt.imread(dir_name + "/" + img_name))
    return np.array(result)

def read_scores(file_name):
    with open(file_name, "r") as file:
        result = []
        for line in file.readlines():
            result.append(float(line))
        return np.array(result)
    
def read_scores_hdai(folder_loc):
    scores = []
    with open("{0}/AIScores.csv".format(folder_loc)) as scores_file:
        scores_file.readline()
        for line in scores_file.readlines():
            splitted = line.split(",")
            scores.append(np.float32(splitted[2]))
    return np.array(scores)

def read_scores_hdxl(folder_loc):
    scores = []
    with open("{0}/names_scores_XL.csv".format(folder_loc)) as scores_file:
        scores_file.readline()
        for line in scores_file.readlines():
            splitted = line.split(",")
            scores.append(np.float32(splitted[1]))
    return np.array(scores)
    
def plot_history(history, validation_freq=1, graph_name="Model loss"):
    if type(history) == type(""):
        with open(history, "rb") as hist_file:
            history = pickle.load(hist_file)
    plt.plot(history['loss'])
    plt.plot(np.arange(len(history['val_loss'])) * validation_freq, history['val_loss'])
    plt.title(graph_name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
    
def generic_batch_generator(pairings, dist_folder="", ref_folder="", n_patches=32, patch_size=32, batch_size=1, ms=False):
    x_res = np.zeros((batch_size, n_patches, patch_size, patch_size, 3))
    y_res = np.zeros((batch_size, n_patches, patch_size, patch_size, 3))
    if ms:
        ps_2 = patch_size // 2
        ps_4 = patch_size // 4
        x_res_ms = np.zeros((batch_size, n_patches, ps_2, ps_2, 3))
        y_res_ms = np.zeros((batch_size, n_patches, ps_2, ps_2, 3))
    prev_ref_im = ""
    batch_idx = 0
    for line in pairings:
        splitted = line.split(',')
        dist_name = splitted[0]
        if dist_folder != "":
            dist_name = "{0}/{1}".format(dist_folder, dist_name)
        ref_name = splitted[1]
        if dist_folder != "":
            ref_name = "{0}/{1}".format(dist_folder, ref_name)
        if ref_name != prev_ref_im:
            y_im = plt.imread(ref_name)
            prev_ref_im = ref_name
            if np.max(y_im) <= 1:
                y_im *= 255
        x_im = plt.imread(dist_name)
        if np.max(x_im) <= 1:
            x_im *= 255

        for i in range(n_patches):
            ul1 = np.random.randint(x_im.shape[0]-patch_size)
            ul2 = np.random.randint(x_im.shape[1]-patch_size)
            x_res[batch_idx][i] = x_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
            y_res[batch_idx][i] = y_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
            if ms:
                x_res_ms[batch_idx][i] = x_im[ul1+ps_4:ul1+3*ps_4, ul2+ps_4:ul2+3*ps_4]
                y_res_ms[batch_idx][i] = y_im[ul1+ps_4:ul1+3*ps_4, ul2+ps_4:ul2+3*ps_4]
        
        batch_idx += 1
        if batch_idx % batch_size == 0:
            batch_idx = 0
            if ms:
                yield({"x32": x_res, "y32": y_res, "x16": x_res_ms, "y16": y_res_ms})
            else:
                yield ({"x": x_res, "y": y_res})
    
def live_batch_generator_ms(image_folder, image_split, scores, orgs, n_p=256):
    refs = scipy.io.loadmat("{0}/refnames_all.mat".format(image_folder))
    dmos = scipy.io.loadmat("{0}/dmos.mat".format(image_folder))
    counter = 0
    img_files = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]
    
    for file_ext in img_files:
        name_list = os.listdir(image_folder + "/" + file_ext)
        name_list = [name for name in name_list if ".bmp" in name]
        for img_name in sorted(name_list, key=lambda x: int(x[3:-4])):
            if not ".bmp" in img_name:
                continue
                
            x_im = plt.imread(image_folder + "/" + file_ext + "/" + img_name).astype(np.float32)# / 255 - 0.5
            y_im = plt.imread("{0}/refimgs/".format(image_folder) + refs["refnames_all"][0][counter][0]).astype(np.float32)# / 255 - 0.5
            score = scores[counter]

            if orgs[counter] == 1:
                counter += 1
                continue

            counter += 1

            if x_im.shape != y_im.shape:
                continue
            
            x_res_32 = np.zeros((n_p,32,32,3))
            y_res_32 = np.zeros((n_p,32,32,3))
            x_res_16 = np.zeros((n_p,16,16,3))
            y_res_16 = np.zeros((n_p,16,16,3))
            for i in range(n_p):
                ul1 = np.random.randint(x_im.shape[0]-32)
                ul2 = np.random.randint(x_im.shape[1]-32)
                x_res_32[i] = x_im[ul1:ul1+32, ul2:ul2+32]
                y_res_32[i] = y_im[ul1:ul1+32, ul2:ul2+32]
                x_res_16[i] = x_im[ul1+8:ul1+24, ul2+8:ul2+24]
                y_res_16[i] = y_im[ul1+8:ul1+24, ul2+8:ul2+24]

            yield ({"x32": np.reshape(x_res_32, (1, n_p, 32, 32, 3)),
                    "y32": np.reshape(y_res_32, (1, n_p, 32, 32, 3)),
                    "x16": np.reshape(x_res_16, (1, n_p, 16, 16, 3)),
                    "y16": np.reshape(y_res_16, (1, n_p, 16, 16, 3))}, np.array(9 - (0.09 * score)).reshape((1,)))
    
def live_batch_generator(image_folder, image_split, scores, orgs, n_p=256, patch_size=32):
    refs = scipy.io.loadmat("{0}/refnames_all.mat".format(image_folder))
    dmos = scipy.io.loadmat("{0}/dmos.mat".format(image_folder))
    counter = 0
    img_files = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]
    
    for file_ext in img_files:
        name_list = os.listdir(image_folder + "/" + file_ext)
        name_list = [name for name in name_list if ".bmp" in name]
        for img_name in sorted(name_list, key=lambda x: int(x[3:-4])):
            if not ".bmp" in img_name:
                continue
                
            x_im = plt.imread(image_folder + "/" + file_ext + "/" + img_name).astype(np.float32)# / 255 - 0.5
            y_im = plt.imread("{0}/refimgs/".format(image_folder) + refs["refnames_all"][0][counter][0]).astype(np.float32)# / 255 - 0.5
            score = scores[counter]

            if orgs[counter] == 1:
                counter += 1
                continue

            counter += 1

            if x_im.shape != y_im.shape:
                continue
            
            x_res = np.zeros((n_p,patch_size,patch_size,3))
            y_res = np.zeros((n_p,patch_size,patch_size,3))
            for i in range(n_p):
                ul1 = np.random.randint(x_im.shape[0]-patch_size)
                ul2 = np.random.randint(x_im.shape[1]-patch_size)
                x_res[i] = x_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
                y_res[i] = y_im[ul1:ul1+patch_size, ul2:ul2+patch_size]

            yield ({"x": np.reshape(x_res, (1, n_p, patch_size, patch_size, 3)),
                    "y": np.reshape(y_res, (1, n_p, patch_size, patch_size, 3))}, np.array(9 - (0.09 * score)).reshape((1,)))
            
def hdai_batch_generator_ms(folder_loc, n_patches=128, image_split=None, endless=False):
    cont = True
    while(cont):
        scores = open("{0}/AIScores.csv".format(folder_loc))
        results = []
        scores.readline()
        lines = np.array(scores.readlines())
        prev_ref_name = ""
        y_im = np.zeros((10,10), dtype=np.float32)
        if not image_split is None:
            lines = lines[image_split]
        for line in lines:
            splitted = line.split(",")
            x_im = plt.imread(folder_loc + "/Distorted" + "/" + splitted[0]).astype(np.float32) * 255# / 255 - 0.5

            ref_im_name = splitted[0].split("_")
            ref_im_name = ref_im_name[0] + "_" + ref_im_name[1] + "_" + ref_im_name[2] + ".png"
            y_im = plt.imread(folder_loc + "/References/" + ref_im_name).astype(np.float32) * 255# / 255 - 0.5

            score = np.float32(splitted[2])

            x_res_32 = np.zeros((n_patches,32,32,3))
            y_res_32 = np.zeros((n_patches,32,32,3))
            x_res_16 = np.zeros((n_patches,16,16,3))
            y_res_16 = np.zeros((n_patches,16,16,3))
            for i in range(n_patches):
                ul1 = np.random.randint(x_im.shape[0]-32)
                ul2 = np.random.randint(x_im.shape[1]-32)
                x_res_32[i] = x_im[ul1:ul1+32, ul2:ul2+32]
                y_res_32[i] = y_im[ul1:ul1+32, ul2:ul2+32]
                x_res_16[i] = x_im[ul1+8:ul1+24, ul2+8:ul2+24]
                y_res_16[i] = y_im[ul1+8:ul1+24, ul2+8:ul2+24]

            yield ({"x32": np.reshape(x_res_32, (1,n_patches,32,32,3)),
                    "y32": np.reshape(y_res_32, (1,n_patches,32,32,3)),
                    "x16": np.reshape(x_res_16, (1,n_patches,16,16,3)),
                    "y16": np.reshape(y_res_16, (1,n_patches,16,16,3))}, np.array(2.25*(score-1)).reshape((1,)))
    
def hdai_batch_generator(folder_loc, n_patches=128, patch_size=32, image_split=None, endless=False, batch_size=1):
    cont = True
    counter = 0
    x_res = np.zeros((batch_size, n_patches,patch_size,patch_size,3), dtype=np.float32)
    y_res = np.zeros((batch_size, n_patches,patch_size,patch_size,3), dtype=np.float32)
    scores_res = np.zeros((batch_size,), dtype=np.float32)
    while(cont):
        scores = open("{0}/AIScores.csv".format(folder_loc))
        results = []
        scores.readline()
        lines = np.array(scores.readlines())
        prev_ref_name = ""
        y_im = np.zeros((10,10), dtype=np.float32)
        if not image_split is None:
            lines = lines[image_split]
        for line in lines:
            splitted = line.split(",")
            x_im = plt.imread(folder_loc + "/Distorted" + "/" + splitted[0]).astype(np.float32) * 255# / 255 - 0.5

            ref_im_name = splitted[0].split("_")
            ref_im_name = ref_im_name[0] + "_" + ref_im_name[1] + "_" + ref_im_name[2] + ".png"
            if prev_ref_name != ref_im_name:
                y_im = plt.imread(folder_loc + "/References/" + ref_im_name).astype(np.float32) * 255# / 255 - 0.5
            prev_ref_name = ref_im_name

            scores_res[counter] = np.float32(splitted[2])
            for i in range(n_patches):
                ul1 = np.random.randint(x_im.shape[0]-patch_size)
                ul2 = np.random.randint(x_im.shape[1]-patch_size)
                x_res[counter][i] = x_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
                y_res[counter][i] = y_im[ul1:ul1+patch_size, ul2:ul2+patch_size]

            counter += 1
            if counter >= batch_size: 
                yield ({"x": x_res, "y": y_res}, 2.25*(scores_res-1))
                counter = 0
        cont = endless
        
def hdxl_batch_generator_ms(folder_loc, n_patches=32, batch_size=2, endless=False):
    cont = True
    counter = 0
    ref_name_counter = -1
    ref_names = ["APPLE_BasketBallScreen_2560x1440p_60_8b_sRGB_444_000.ppm",
                 "ARRI_PublicUniversity_2880x1620p_24_8b_bt709_444_0000.ppm",
                 "BIKE_2048x2560_8b_RGB.ppm",
                 "CAFE_2048x2560_8b_RGB.ppm",
                 "FemaleStripedHorseFly_1920x1080_8b.ppm",
                 "WOMAN_2048x2560_8b_RGB.ppm",
                 "p06.ppm"
                 ]
    x_res = np.zeros((batch_size, n_patches,32,32,3), dtype=np.float32)
    y_res = np.zeros((batch_size, n_patches,32,32,3), dtype=np.float32)
    x_res_16 = np.zeros((batch_size, n_patches,16,16,3), dtype=np.float32)
    y_res_16 = np.zeros((batch_size, n_patches,16,16,3), dtype=np.float32)
    scores_res = np.zeros((batch_size,), dtype=np.float32)
    while(cont):
        counter = 0
        ref_name_counter = -1
        scores = open("{0}/names_scores_XL.csv".format(folder_loc), encoding="utf-8")
        results = []
        scores.readline()
        lines = np.array(scores.readlines())
        prev_ref_name = "meaningless_name"
        y_im = np.zeros((10,10), dtype=np.float32)
        for line in lines:
            splitted = line.split(",")
            x_im = np.array(Image.open(folder_loc + "/Distorted" + "/" + splitted[0])).astype(np.float32)

            if prev_ref_name[:3] != line[:3]:
                ref_name_counter += 1
                prev_ref_name = ref_names[ref_name_counter]
                y_im = np.array(Image.open(folder_loc + "/References/" + prev_ref_name)).astype(np.float32)

            if x_im.shape != y_im.shape:
                print("Shape mismatch " + line + " : " + prev_ref_name)
                print(counter)
            scores_res[counter % batch_size] = np.float32(splitted[1])
            for i in range(n_patches):
                ul1 = np.random.randint(x_im.shape[0]-32)
                ul2 = np.random.randint(x_im.shape[1]-32)
                x_res[counter % batch_size][i] = x_im[ul1:ul1+32, ul2:ul2+32]
                y_res[counter % batch_size][i] = y_im[ul1:ul1+32, ul2:ul2+32]
                x_res_16[counter % batch_size][i] = x_im[ul1+8:ul1+24, ul2+8:ul2+24]
                y_res_16[counter % batch_size][i] = y_im[ul1+8:ul1+24, ul2+8:ul2+24]

            counter += 1
            if counter % batch_size == 0: 
                yield ({"x32": x_res, "y32": y_res, "x16": x_res_16, "y16": y_res_16}, 2.25*(scores_res-1))
        cont = endless
        
def hdxl_batch_generator(folder_loc, n_patches=32, patch_size=32, batch_size=2, endless=False):
    cont = True
    counter = 0
    ref_name_counter = -1
    ref_names = ["APPLE_BasketBallScreen_2560x1440p_60_8b_sRGB_444_000.ppm",
                 "ARRI_PublicUniversity_2880x1620p_24_8b_bt709_444_0000.ppm",
                 "BIKE_2048x2560_8b_RGB.ppm",
                 "CAFE_2048x2560_8b_RGB.ppm",
                 "FemaleStripedHorseFly_1920x1080_8b.ppm",
                 "WOMAN_2048x2560_8b_RGB.ppm",
                 "p06.ppm"
                 ]
    x_res = np.zeros((batch_size, n_patches,patch_size,patch_size,3), dtype=np.float32)
    y_res = np.zeros((batch_size, n_patches,patch_size,patch_size,3), dtype=np.float32)
    scores_res = np.zeros((batch_size,), dtype=np.float32)
    while(cont):
        counter = 0
        ref_name_counter = -1
        scores = open("{0}/names_scores_XL.csv".format(folder_loc), encoding="utf-8")
        results = []
        scores.readline()
        lines = np.array(scores.readlines())
        prev_ref_name = "meaningless_name"
        y_im = np.zeros((10,10), dtype=np.float32)
        for line in lines:
            splitted = line.split(",")
            x_im = np.array(Image.open(folder_loc + "/Distorted" + "/" + splitted[0])).astype(np.float32)

            if prev_ref_name[:3] != line[:3]:
                ref_name_counter += 1
                prev_ref_name = ref_names[ref_name_counter]
                y_im = np.array(Image.open(folder_loc + "/References/" + prev_ref_name)).astype(np.float32)

            if x_im.shape != y_im.shape:
                print("Shape mismatch " + line + " : " + prev_ref_name)
                print(counter)
            scores_res[counter % batch_size] = np.float32(splitted[1])
            for i in range(n_patches):
                ul1 = np.random.randint(x_im.shape[0]-patch_size)
                ul2 = np.random.randint(x_im.shape[1]-patch_size)
                x_res[counter % batch_size][i] = x_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
                y_res[counter % batch_size][i] = y_im[ul1:ul1+patch_size, ul2:ul2+patch_size]

            counter += 1
            if counter % batch_size == 0: 
                yield ({"x": x_res, "y": y_res}, 2.25*(scores_res-1))
        cont = endless
        
def csiq_batch_generator_ms(csiq_loc, n_patches=32, batch_size=10, endless=False):
    counter = 0
    cont = True
    wb = xlrd.open_workbook("{0}/csiq.DMOS.xlsx".format(csiq_loc))
    sheet = wb.sheet_by_index(5)
    old_im_name = ""
    dist_folders = ["", "awgn", "jpeg", "jpeg2000", "fnoise", "blur", "contrast"]
    im_mods = ["", "AWGN", "JPEG", "jpeg2000", "fnoise", "BLUR", "contrast"]
    x_res = np.zeros((batch_size, n_patches, 32, 32, 3))
    y_res = np.zeros((batch_size, n_patches, 32, 32, 3))
    x_res_16 = np.zeros((batch_size, n_patches, 16, 16, 3))
    y_res_16 = np.zeros((batch_size, n_patches, 16, 16, 3))
    scores = np.zeros((batch_size,))
    while cont:
        im_name = sheet.cell_type(counter + 4, 3)
        if im_name == 2:
            im_name = str(int(sheet.cell_value(counter + 4, 3)))
        else:
            im_name = sheet.cell_value(counter + 4, 3)
        scores[counter % batch_size] = sheet.cell_value(counter + 4, 8)
        dist_folder = dist_folders[int(sheet.cell_value(counter + 4, 4))]
        im_mod = im_mods[int(sheet.cell_value(counter + 4, 4))]
        dist_level = str(int(sheet.cell_value(counter + 4, 6)))
        im_file = "{0}/distorted_images/{1}/{2}.{3}.{4}.png".format(csiq_loc, dist_folder, im_name, im_mod, dist_level)
        x_im = plt.imread(im_file) * 255.0
        if im_name != old_im_name:
            y_im = plt.imread("{0}/reference_images/{1}.png".format(csiq_loc, im_name)) * 255.0
            old_im_name = im_name
        
        for i in range(n_patches):
            ul1 = np.random.randint(x_im.shape[0]-32)
            ul2 = np.random.randint(x_im.shape[1]-32)
            x_res[counter % batch_size][i] = x_im[ul1:ul1+32, ul2:ul2+32]
            y_res[counter % batch_size][i] = y_im[ul1:ul1+32, ul2:ul2+32]
            x_res_16[counter % batch_size][i] = x_im[ul1+8:ul1+24, ul2+8:ul2+24]
            y_res_16[counter % batch_size][i] = y_im[ul1+8:ul1+24, ul2+8:ul2+24]

        counter += 1
        if counter % batch_size == 0:
            yield ({"x32": x_res, "y32": y_res, "x16": x_res_16, "y16": y_res_16}, scores)
            
        if counter >= 866:
            if endless:
                counter = 0
            else:
                cont = False
        
def csiq_batch_generator(csiq_loc, n_patches=32, patch_size=32, batch_size=10, endless=False):
    counter = 0
    cont = True
    wb = xlrd.open_workbook("{0}/csiq.DMOS.xlsx".format(csiq_loc))
    sheet = wb.sheet_by_index(5)
    old_im_name = ""
    dist_folders = ["", "awgn", "jpeg", "jpeg2000", "fnoise", "blur", "contrast"]
    im_mods = ["", "AWGN", "JPEG", "jpeg2000", "fnoise", "BLUR", "contrast"]
    x_res = np.zeros((batch_size, n_patches, patch_size, patch_size, 3))
    y_res = np.zeros((batch_size, n_patches, patch_size, patch_size, 3))
    scores = np.zeros((batch_size,))
    while cont:
        im_name = sheet.cell_type(counter + 4, 3)
        if im_name == 2:
            im_name = str(int(sheet.cell_value(counter + 4, 3)))
        else:
            im_name = sheet.cell_value(counter + 4, 3)
        scores[counter % batch_size] = sheet.cell_value(counter + 4, 8)
        dist_folder = dist_folders[int(sheet.cell_value(counter + 4, 4))]
        im_mod = im_mods[int(sheet.cell_value(counter + 4, 4))]
        dist_level = str(int(sheet.cell_value(counter + 4, 6)))
        im_file = "{0}/distorted_images/{1}/{2}.{3}.{4}.png".format(csiq_loc, dist_folder, im_name, im_mod, dist_level)
        x_im = plt.imread(im_file) * 255.0
        if im_name != old_im_name:
            y_im = plt.imread("{0}/reference_images/{1}.png".format(csiq_loc, im_name)) * 255.0
            old_im_name = im_name
        
        for i in range(n_patches):
            ul1 = np.random.randint(x_im.shape[0]-patch_size)
            ul2 = np.random.randint(x_im.shape[1]-patch_size)
            x_res[counter % batch_size][i] = x_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
            y_res[counter % batch_size][i] = y_im[ul1:ul1+patch_size, ul2:ul2+patch_size]

        counter += 1
        if counter % batch_size == 0:
            yield ({"x": x_res, "y": y_res}, scores)
            
        if counter >= 866:
            if endless:
                counter = 0
            else:
                cont = False
        
def tid_batch_generator_ms(tid_loc, reference_images, mos_scores, img_names, shuffle, image_split=np.arange(25), n_patches=32, patch_size=32, batch_size=10):
    counter = 0
    cont = True
    while cont:
        x_res_32 = np.zeros((batch_size, n_patches, 32, 32, 3))
        y_res_32 = np.zeros((batch_size, n_patches, 32, 32, 3))
        x_res_16 = np.zeros((batch_size, n_patches, 16, 16, 3))
        y_res_16 = np.zeros((batch_size, n_patches, 16, 16, 3))
        scores = np.zeros((batch_size,))
        for bat_idx in range(batch_size):
            img_idx = random.choice(image_split)
            dist_idx = random.randint(0,119)
            if not shuffle:
                img_idx = image_split[int(counter / 120)]
                dist_idx = counter % 120
            counter += 1

            y_im = reference_images[img_idx]
            if dist_idx < 120:
                x_im = plt.imread(tid_loc + "/distorted_images/" + (img_names[120 * img_idx + dist_idx])).astype(np.float32)# / 255 - 0.5
                scores[bat_idx] = mos_scores[120 * img_idx + dist_idx].reshape((1))
            else:
                x_im = y_im
                scores[bat_idx] = np.array([9])

            for i in range(n_patches):
                ul1 = np.random.randint(x_im.shape[0]-32)
                ul2 = np.random.randint(x_im.shape[1]-32)
                x_res_32[bat_idx][i] = x_im[ul1:ul1+32, ul2:ul2+32]
                y_res_32[bat_idx][i] = y_im[ul1:ul1+32, ul2:ul2+32]
                x_res_16[bat_idx][i] = x_im[ul1+8:ul1+24, ul2+8:ul2+24]
                y_res_16[bat_idx][i] = y_im[ul1+8:ul1+24, ul2+8:ul2+24]

            if not shuffle and counter >= image_split.shape[0] * 120:
                cont = False
            
        yield ({"x32": x_res_32, "y32": y_res_32, "x16": x_res_16, "y16": y_res_16}, scores)
    
def tid_batch_generator(tid_loc, reference_images, mos_scores, img_names, shuffle, image_split=np.arange(25), n_patches=32, patch_size=32, batch_size=10, full_images=False):
    counter = 0
    cont = True
    while cont:
        scores = np.zeros((batch_size,))
        x_res = np.zeros((batch_size, n_patches, patch_size, patch_size, 3))
        y_res = np.zeros((batch_size, n_patches, patch_size, patch_size, 3))
        distorted_img_dir = "{0}/distorted_images".format(tid_loc)
        for batch_idx in range(batch_size):
            img_idx = random.choice(image_split)
            dist_idx = random.randint(0,125)
            if not shuffle:
                img_idx = image_split[int(counter / 120)]
                dist_idx = counter % 120
            counter += 1

            y_im = reference_images[img_idx]
            if dist_idx < 120:
                x_im = plt.imread(distorted_img_dir + "/" + (img_names[120 * img_idx + dist_idx])).astype(np.float32)# / 255 - 0.5
                scores[batch_idx] = mos_scores[120 * img_idx + dist_idx].reshape((1))
            else:
                x_im = y_im
                scores[batch_idx] = np.float32(9)

            if full_images:
                yield ({"x": x_im, "y": y_im}, scores[batch_idx])
            for i in range(n_patches):
                ul1 = np.random.randint(x_im.shape[0]-patch_size)
                ul2 = np.random.randint(x_im.shape[1]-patch_size)
                x_res[batch_idx][i] = x_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
                y_res[batch_idx][i] = y_im[ul1:ul1+patch_size, ul2:ul2+patch_size]
        if not shuffle and counter >= len(image_split) * 120:
            cont = False
        if not full_images:
            yield ({"x": x_res, "y": y_res}, scores)
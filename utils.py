"""
Created on Moon Festival + 1 2021

utils

@author: Ray
"""
import os, glob, random, cv2
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing import image

def load_path(folder):
    paths = []
    if len(glob.glob(folder + r'\**.png')) != 0:
        paths = glob.glob(folder + r'\**.png')

        if len(glob.glob(folder + r'\**.jpg')) != 0:
            paths.extend(glob.glob(folder + r'\**.jpg'))

    elif len(glob.glob(folder + r'\**.png')) == 0:
        if len(glob.glob(folder + r'\**.jpg')) != 0:
            paths = glob.glob(folder + r'\**.jpg')
    else: 
        pass

    return paths

def fill_data(A_path, B_path):
    if len(B_path) < len(A_path):
        num = len(A_path) - len(B_path)
        fill = [random.randint(0, len(B_path)-1) for _ in range(num)]
        for i in fill: B_path.append(B_path[i]) 

    elif len(B_path) > len(A_path):
        num = len(B_path) - len(A_path)
        fill = [random.randint(0, len(A_path)-1) for _ in range(num)]
        for i in fill: A_path.append(A_path[i]) 
    else:
        pass
    return A_path, B_path

def random_crop(inputs):
    ranges = random.uniform(0.5, 1)
    outputs = image.random_zoom(inputs, zoom_range=(ranges,ranges), row_axis=0, col_axis=1, channel_axis=2)
    outputs = cv2.resize(outputs, (128, 128), interpolation=cv2.INTER_LINEAR)
    return outputs

def load_data(A_path, B_path):
    a_imgs = []
    b_imgs = []
    for i in range(len(A_path)):
        a_img = np.array(image.load_img(A_path[i]))
        b_img = np.array(image.load_img(B_path[i]))
        a_img = random_crop(a_img)
        b_img = random_crop(b_img)
        a_img = (a_img / 127.5) - 1
        b_img = (b_img / 127.5) - 1
        a_imgs.append(a_img)
        b_imgs.append(b_img)
    return a_imgs, b_imgs

def load_test_data(A_path):

    g_img = np.array(image.load_img(A_path, target_size=(128,128)))
    g_img = (g_img / 127.5) - 1
    g_img = np.expand_dims(g_img, axis=0)

    return list(g_img)

def resize(o_img_path, g_img):
    o_img = np.array(image.load_img(o_img_path))
    m, n, _ = o_img.shape
    g_img = tf.image.resize(g_img, [m, n])

    return g_img

def save_data(im, images, save_path=None):
    if save_path == None:
        save_path = os.getcwd() + '\\result'
        for i in range(len(images)):
            os.mkdir(os.getcwd() + '\\result') if os.path.exists(os.getcwd() + '\\result') == False else None
            im[i].save(save_path + '\\' + os.path.basename(images[i]))

    else:
        for i in range(len(images)):
            im[i].save(save_path + '\\' + os.path.basename(images[i])) if os.path.exists(save_path) == True else sys.exit('Dir not exist')

    return save_path
# make a prediction for a new image.
from collections import Counter
from os import listdir
from os.path import isfile, join
# from random import shuffle, random

import cv2
import emnist
import h5py
import numpy as np
from joblib.numpy_pickle_utils import xrange
from numpy import argmax, random
# import tensorflow.keras as keras
from keras.utils.image_utils import load_img, img_to_array
from keras.models import load_model
import os
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from extra_keras_datasets import emnist

from emnist import extract_training_samples, extract_test_samples
# from numpy.random import shuffle
# from numpy.random.mtrand import random
from sklearn.model_selection import train_test_split
from sklearn.externals._pilutil import imresize

# from src.transformation import character_curated, train_rigth_borders, d_num_cases, train_left_borders
from sklearn.utils import shuffle, random

digit_location = "../resources/out/"
path = 'C:/Users/Itay/Downloads/curated/curated/'

d_num_cases = ()


def prepare_data_char_subset(X, y):
    X_select = X
    y_select = y

    # Shuffle
    X_select, y_select = shuffle(X_select, y_select, random_state=0)

    # Separate train test
    X_train, X_test, y_train, y_test = train_test_split(X_select, y_select, test_size=0.20, random_state=42)
    print(X_train.shape)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    print(X_train.shape)
    # print('Train shape: ', X_train.shape, y_train.shape)
    # print('Test shape: ', X_test.shape, y_test.shape)
    # print('Num classes: ', len(set(y_train)))
    # print('Classes: ', set(y_train))

    return X_train, y_train, X_test, y_test


def move_characters(X, y):
    # global imresize
    X_moved = []
    y_moved = []

    for i, y_value in enumerate(y):
        y_char = chr(y_value)
        if y_char in ['a', 'c', 'e', 'i', 'm', 'n', 'o', 'r', 's', 'u', 'v', 'w', 'x', 'z']:
            # Resize to 40x40 and place botton, medium and top
            img1 = imresize(X[i, 0, :, :], (46, 46))

            img_center = np.zeros((64, 64))
            img_center[9:55, 9:55] = img1
            X_moved += [img_center]
            y_moved += [y_value]

            img_top = np.zeros((64, 64))
            img_top[:46, 9:55] = img1
            X_moved += [img_top]
            y_moved += [y_value]

            img_botton = np.zeros((64, 64))
            img_botton[18:, 9:55] = img1
            X_moved += [img_botton]
            y_moved += [y_value]

        elif y_char in ['g', 'j', 'p', 'q', 'y']:
            # Resize to 52x52 and place botton
            img1 = imresize(X[i, 0, :, :], (46, 46))
            img_botton = np.zeros((64, 64))
            img_botton[18:, 9:55] = img1
            X_moved += [img_botton]
            y_moved += [y_value]

        elif y_char in ['b', 'd', 'f', 'h', 'k', 'l', 't', 'A', 'B', 'C', 'D', 'E',
                        'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                        'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!', '\"', '#',
                        '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
                        '<', '=', '>', '?', '@', '[', ']', '^', '_', '`', '{',
                        '|', '}', '~']:
            # Resize to 52x52 and place on top
            img1 = imresize(X[i, 0, :, :], (46, 46))
            img_top = np.zeros((64, 64))
            img_top[:46, 9:55] = img1
            X_moved += [img_top]
            y_moved += [y_value]

    X_moved = np.array(X_moved, dtype=np.float16)
    y_moved = np.array(y_moved)
    X_moved = np.reshape(X_moved, (X_moved.shape[0], 1, X_moved.shape[1], X_moved.shape[2]))
    return X_moved, y_moved


# create 64x8 borders left and right
def obtain_borders(img, threshold=0.):
    # Identify border of the letter
    border_left = img[:, :8]
    left_position = 0
    for i in xrange(0, 56, 1):
        # print(i)
        if np.max(img[:, i]) > threshold:
            left_position = i
            break
    rigth_position = 64
    border_rigth = img[:, 56:]
    for i in xrange(63, 8, -1):
        if np.max(img[:, i]) > threshold:
            rigth_position = i
            break
    return img[:, left_position:left_position + 8], img[:, rigth_position - 8:rigth_position], img[:,
                                                                                               left_position:rigth_position]


def generate_borders_images(X, y, train_right_borders, train_left_borders):
    X_left = []
    y_left = []
    X_rigth = []
    y_rigth = []
    for i, y_value in enumerate(y):
        y_char = y_value
        # n_images = int(300 / 2)
        n_images = int(10000 / d_num_cases[y_char])  # balanced sample
        print(y_char, len(d_num_cases))
        if y_char < len(d_num_cases):
            n_images = int(30000 / (d_num_cases[y_char]))  # Reduce by 2 the balanced sample to reduce sample

        for j in xrange(int(n_images / 2)):  # two images each iteration
            n_borders = int(np.random.uniform(0, X.shape[0]))
            img_left = np.copy(X[i, 0, :, :])
            img_left[:, :5] = train_right_borders[n_borders, :, -5:]  # put the rigth side of the rigth border
            X_left += [img_left]
            y_left += [y_value]

            img_rigth = np.copy(X[i, 0, :, :])
            img_rigth[:, -5:] = train_left_borders[n_borders, :, :5]  # put the left side of the left border
            X_rigth += [img_rigth]
            y_rigth += [y_value]

    X_left = np.array(X_left, dtype=np.float16)
    X_left = np.reshape(X_left, (X_left.shape[0], 1, X_left.shape[1], X_left.shape[2]))
    y_left = np.array(y_left)
    X_rigth = np.array(X_rigth, dtype=np.float16)
    X_rigth = np.reshape(X_rigth, (X_rigth.shape[0], 1, X_rigth.shape[1], X_rigth.shape[2]))
    y_rigth = np.array(y_rigth)
    return X_left, y_left, X_rigth, y_rigth


def train_chars():
    X = []
    y = []
    character_curated = [ord(c) for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ']
    # character_curated = [ord(c) for c in
    #                      '!\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~']
    for i in character_curated:
        path_img = path + str(i) + '/'
        for file_name in [f for f in listdir(path_img) if isfile(join(path_img, f))]:
            img = cv2.imread(path_img + file_name, 0)
            # img = cv2.resize(img,(32, 32), interpolation = cv2.INTER_AREA)
            X += [img]
            y += [i]

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.uint8)
    X_train, y_train, X_test, y_test = prepare_data_char_subset(X, y)
    # X_train_moved, y_train_moved = move_characters(X_train[4347:4449], y_train[4347:4449])

    # c = Counter(y_train)
    # d_num_cases = dict(c.most_common())
    # # print(d_num_cases)
    #
    # X_train_moved, y_train_moved = move_characters(X_train, y_train)
    # # print(X_train_moved.shape, y_train_moved.shape)
    #
    # X_test_moved, y_test_moved = move_characters(X_test, y_test)
    # # print(X_test_moved.shape, y_test_moved.shape)
    # train_left_borders = []
    # train_rigth_borders = []
    # # border_left, border_rigth, char_adjusted = obtain_borders(X_train[0, 0, :, :])
    #
    # train_img_no_borders = []
    # for i, y_value in enumerate(y_train):
    #     # l, r, c = obtain_borders(X_train[i, 0, :, :], treshold=0.)
    #     l, r, c = (0, 64, 32)
    #     train_left_borders += [l]
    #     train_rigth_borders += [r]
    #     train_img_no_borders += [c]
    # train_left_borders = np.array(train_left_borders)
    # train_right_borders = np.array(train_rigth_borders)
    #
    # # test_left_borders = []
    # # test_right_borders = []
    # # test_img_no_borders = []
    # # for i, y_value in enumerate(y_test):
    # #     l, r, c = obtain_borders(X_test[i, 0, :, :], treshold=0.)
    # #     test_left_borders += [l]
    # #     test_right_borders += [r]
    # #     test_img_no_borders += [c]
    # # test_left_borders = np.array(test_left_borders)
    # # test_rigth_borders = np.array(test_right_borders)
    # #
    # # X_trn_borders_left = []
    # # X_trn_borders_rigth = []
    # # y_trn_borders_left = []
    # # y_trn_borders_rigth = []
    #
    # # X_train_left, y_train_left, X_train_rigth, y_train_rigth = (X_train[300:315], y_train[300:315], train_right_borders, train_left_borders)
    #
    # X_train_left, y_train_left, X_train_rigth, y_train_rigth = (
    #     X_train, y_train, train_right_borders, train_left_borders)
    # print(X_train_left.shape, y_train_left.shape, X_train_rigth.shape, y_train_rigth.shape)
    #
    # X_test_left, y_test_left, X_test_rigth, y_test_rigth = (X_test, y_test, train_right_borders, train_left_borders)
    # print(X_test_left.shape, y_test_left.shape, X_test_rigth.shape, y_test_rigth.shape)

    # Combine all sources

    # X_train_aug = np.concatenate((X_train, X_train_moved, X_train_left, X_train_rigth), axis=0)
    # y_train_aug = np.concatenate((y_train, y_train_moved, y_train_left, y_train_rigth), axis=0)
    #
    # X_test_aug = np.concatenate((X_test, X_test_moved, X_test_left, X_test_rigth), axis=0)
    # y_test_aug = np.concatenate((y_test, y_test_moved, y_test_left, y_test_rigth), axis=0)
    #
    # print(X_train_aug.shape, y_train_aug.shape, X_train_aug.shape, y_train_aug.shape)
    # print(X_test_aug.shape, y_test_aug.shape, X_test_aug.shape, y_test_aug.shape)

    # Shuffle train data
    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    # standarize
    X_train = (X_train - 128.) / 255.
    X_test = (X_test - 128.) / 255.

    print("saving")

    hdf5_f = h5py.File("../models/only_characters_upcase_lowcase_64x64.hdf5", mode='a')

    if "X_train" in hdf5_f:
        del hdf5_f["X_train"]
    hdf5_f.create_dataset("X_train", data=X_train)

    if "y_train" in hdf5_f:
        del hdf5_f["y_train"]
    hdf5_f.create_dataset("y_train", data=y_train)

    if "X_test" in hdf5_f:
        del hdf5_f["X_test"]
    hdf5_f.create_dataset("X_test", data=X_test)

    if "y_test" in hdf5_f:
        del hdf5_f["y_test"]
    hdf5_f.create_dataset("y_test", data=y_test)

    hdf5_f.close()


# def bytes_to_int(byte_data):
#     return int.from_bytes(byte_data, 'big')
#
#
# def read_images(filename, num_max_img=None):
#     images = []
#     with open(filename, 'rb') as f:
#         _ = f.read(4)  # magic number
#         # n_images = bytes_to_int(f.read(4))
#         n_images = f.read(4)
#         if num_max_img:
#             n_images = num_max_img
#         n_rows = bytes_to_int(f.read(4))
#         n_columns = bytes_to_int(f.read(4))
#         # n_rows = f.read(4)
#         # n_columns = f.read(4)
#         for image_idx in range(n_images):
#             image = []
#             for row_idx in range(n_rows):
#                 row = []
#                 for col_idx in range(n_columns):
#                     # pixel = bytes_to_int(f.read(1))
#                     pixel = (f.read(1))
#                     row.append(pixel)
#                 image.append(row)
#             images.append(image)
#     return images
#
#
# def read_labels(filename, num_max_labels=None):
#     labels = []
#     with open(filename, 'rb') as f:
#         _ = f.read(4)  # magic number
#         # n_labels = bytes_to_int(f.read(4))
#         n_labels = (f.read(4))
#         if num_max_labels:
#             n_labels = num_max_labels
#         for label_idx in range(n_labels):
#             # label = bytes_to_int(f.read(1))
#             label = (f.read(1))
#             labels.append(label)
#     return labels

def regex_filename(filename):
    split_filename = filename[:-5].split("_")
    return int(split_filename[0]), int(split_filename[1])


def write_to_file(chars):
    txt_file = open('../output/output_text.txt', 'w')
    prev_line = 0
    prev_letter = 0
    txt = ""
    # prev_char = ''
    for c in chars:
        filename, char = c
        line, letter = regex_filename(filename)
        if prev_line < line:
            txt += "\n"
        if prev_letter + 1 < letter:
            txt += " "
        txt += char
        prev_line = line
        prev_letter = letter
    for s in range(len(txt)):
        if txt[s] == 'O' and \
                (s - 1 >= 0 and s + 1 < len(txt) and (
                        txt[s - 1] != ' ' and '0' <= txt[s - 1] <= '9' or txt[s + 1] != ' ' and '0' <= txt[
                    s + 1] <= '9')
                 or (s - 2 >= 0 and s + 2 < len(txt) and (
                                txt[s - 2] != ' ' and '0' <= txt[s - 2] <= '9' or txt[s + 2] != ' ' and '0' <= txt[
                            s + 2] <= '9'))):
            txt = txt.replace(txt[s], "0")

        if txt[s] == '0' and \
                (s - 1 >= 0 and s + 1 < len(txt) and (
                        txt[s - 1] != ' ' and 'A' <= txt[s - 1] <= 'Z' or txt[s + 1] != ' ' and 'A' <= txt[
                    s + 1] <= 'Z')
                 or (s - 2 >= 0 and s + 2 < len(txt) and (
                                txt[s - 2] != ' ' and 'A' <= txt[s - 2] <= 'Z' or txt[s + 2] != ' ' and 'A' <= txt[
                            s + 2] <= 'Z'))):
            txt = txt.replace(txt[s], "O")

        if txt[s] == 'I' and \
                (s - 1 >= 0 and s + 1 < len(txt) and (
                        txt[s - 1] != ' ' and '0' <= txt[s - 1] <= '9' or txt[s + 1] != ' ' and '0' <= txt[
                    s + 1] <= '9')
                 or (s - 2 >= 0 and s + 2 < len(txt) and (
                                txt[s - 2] != ' ' and '0' <= txt[s - 2] <= '9' or txt[s + 2] != ' ' and '0' <= txt[
                            s + 2] <= '9'))):
            txt = txt.replace(txt[s], "1")

        if txt[s] == '1' and \
                (s - 1 >= 0 and s + 1 < len(txt) and (
                        txt[s - 1] != ' ' and 'A' <= txt[s - 1] <= 'Z' or txt[s + 1] != ' ' and 'A' <= txt[
                    s + 1] <= 'Z')
                 or (s - 2 >= 0 and s + 2 < len(txt) and (
                                txt[s - 2] != ' ' and 'A' <= txt[s - 2] <= 'Z' or txt[s + 2] != ' ' and 'A' <= txt[
                            s + 2] <= 'Z'))):
            txt = txt.replace(txt[s], "I")
    txt_file.write(txt)
    txt_file.close()


def get_ascii(char):
    if 0 <= char <= 9:
        return chr(48 + char)
    elif 10 <= char <= 36:
        return chr(55 + char)
    return chr(char)


# load train and test dataset
def load_dataset():
    X = []
    y = []
    # character_curated = [ord(c) for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ']
    character_curated = [ord(c) for c in
                         '!\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~']
    for i in character_curated:
        path_img = path + str(i) + '/'
        for file_name in [f for f in listdir(path_img) if isfile(join(path_img, f))]:
            img = cv2.imread(path_img + file_name, 0)
            # img = cv2.resize(img,(32, 32), interpolation = cv2.INTER_AREA)
            X += [img]
            y += [i]

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.uint8)
    X_train, y_train, X_test, y_test = prepare_data_char_subset(X, y)

    # Shuffle train data
    # X_train, y_train = shuffle(X_train, y_train, random_state=0)
    #
    # # standarize
    # X_train = (X_train - 128.) / 255.
    # X_test = (X_test - 128.) / 255.

    trainX = X_train.reshape((X_train.shape[0], 64, 64, 1))
    testX = X_test.reshape((X_test.shape[0], 64, 64, 1))
    # one hot encode target values
    trainY = to_categorical(y_train)
    testY = to_categorical(y_test)

    # # load dataset
    # (trainX, trainY), (testX, testY) = mnist.load_data()
    # # reshape dataset to have a single channel
    # trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    # testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # # one hot encode target values
    # trainY = to_categorical(trainY)
    # testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    # model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# run the test harness for evaluating a model
def run_test_harness1():
    # load dataset
    # (trainX, trainY), (testX, testY) = emnist.load_data(type='balanced')

    # trainX, trainY = extract_training_samples('balanced')
    # testX, testY = extract_test_samples('balanced')

    # hdf5_f = h5py.File("../models/only_characters_upcase_lowcase_64x64.hdf5", mode='r')
    #
    # X = hdf5_f["X_train"]
    # y = hdf5_f["y_train"]
    # print(X.shape, y.shape)
    #
    # trainX = np.copy(X)
    # trainY = np.copy(y)
    #
    # X = hdf5_f["X_test"]
    # y = hdf5_f["y_test"]
    # print(X.shape, y.shape)
    # testX = np.copy(X)
    # testY = np.copy(y)
    #
    # hdf5_f.close()

    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data

    trainX, testX = prep_pixels(trainX, testX)

    # trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    # testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # trainY = to_categorical(trainY)
    # testY = to_categorical(testY)
    # define model
    model = define_model()
    model.add(Dense(trainY.shape[1], activation='softmax'))

    # fit model
    model.fit(trainX, trainY, epochs=100, batch_size=512, verbose=1)  # 10, 32, 0
    # save model
    model.save('../models/new_model_%_100_512_2.h5')


# run the test harness for evaluating a model
def run_test_harness2():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # load model
    model = load_model('model_57%_14_64.h5')
    # evaluate model on test dataset
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))


# load and prepare the image
def load_image(filename):
    # load the image
    # img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    img = load_img(filename, color_mode="grayscale", target_size=(64, 64))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    # img = img.reshape(1, 28, 28, 1)
    img = img.reshape(1, 64, 64, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# load an image and predict the class
def run_example():
    # load the image
    directory = "../resources/out"
    chars = []
    for filename in os.listdir(directory):
        img = load_image(digit_location + filename)
        # load model
        model = load_model('../models/new_model_%_100_512_2.h5')

        # predict the class
        predict_value = model.predict(img)
        digit = argmax(predict_value)
        # chars.append((filename, get_ascii(digit)))
        chars.append((filename, chr(digit)))
        # print(filename, get_ascii(digit), digit)
        print(filename, chr(digit), digit)
    write_to_file(chars)
    print()


if __name__ == '__main__':
    # make dataset from pictures
    # train_chars()

    # entry point, run the test harness
    # run_test_harness1()

    # entry point, run the example
    run_example()

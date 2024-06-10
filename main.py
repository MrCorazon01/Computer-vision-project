import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import os
import cv2

import re

import string
import time

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

char_encoding = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
max_len_label = 0

def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_encoding.index(char))
        except:
            print(char)
        
    return dig_lst

valid_path = []
valid_label = []
valid_label_enc = []
valid_input_len = []
valid_label_len = []

with open("/content/RWTH.iam_word_gt_final.valid.thresh") as valid_list:
  for value in valid_list:
    valid_path.append(value.split(",")[0].strip()+".png")
    label = value.split(" ",1)[1].strip()
    valid_label.append(label)
    valid_label_enc.append(encode_to_labels(label))
    valid_label_len.append(len(label))
    valid_input_len.append(31)


max_len_label = max(max_len_label,len(max(valid_label, key=len)))
# print(max_len_label)

train_path = []
train_label = []
train_label_enc = []
train_input_len = []
train_label_len = []

with open("/content/RWTH.iam_word_gt_final.train.thresh") as train_list:
  for value in train_list:
    train_path.append(value.split(",")[0].strip()+".png")
    label = value.split(" ",1)[1].strip()
    train_label.append(label)
    train_label_enc.append(encode_to_labels(label))
    train_label_len.append(len(label))
    train_input_len.append(31)

print(train_label[:5])

max_len_label = max(max_len_label,len(max(train_label, key=len)))
# print(max_len_label)

test_path = []
test_label = []
with open("/content/RWTH.iam_word_gt_final.test.thresh") as test_list:
  for value in test_list:
    test_path.append(value.split(",")[0].strip()+".png")
    test_label.append(value.split(" ",1)[1].strip())

# print(test_path)

max_len_label = max(max_len_label,len(max(test_label, key=len)))
# print(max_len_label)


train_padded_txt = pad_sequences(train_label_enc, maxlen=max_len_label, padding='post', value = len(char_encoding))
valid_padded_txt = pad_sequences(valid_label_enc, maxlen=max_len_label, padding='post', value = len(char_encoding))

train_img = []
valid_img = []
test_img = []
all_paths = {}

for root, dirs, files in os.walk("words/"):
  for file in files:
    all_paths[file]=os.path.join(root, file)


for file in train_path:
  if file in all_paths:
    input_img = cv2.cvtColor(cv2.imread(all_paths[file]),cv2.COLOR_BGR2GRAY)
    # input_img = cv2.imread(all_paths[file])
    # print(input_img.shape, type(input_img))
    # print(input_img)
    input_img = cv2.resize(input_img, (128,32))
    # print(input_img.shape)
    # print(input_img)
    input_img = np.expand_dims(input_img, axis=2)
    # input_img = cv2.resize(input_img, (128,32))
    # print(input_img.shape)
    input_img = input_img/255
    # print(input_img)
    train_img.append(input_img)

for file in valid_path:
  if file in all_paths:
    input_img = cv2.cvtColor(cv2.imread(all_paths[file]),cv2.COLOR_BGR2GRAY)
    input_img = cv2.resize(input_img, (128,32))
    input_img = np.expand_dims(input_img, axis=2)
    input_img = input_img/255
    valid_img.append(input_img)

for file in test_path:
  if file in all_paths:
    input_img = cv2.cvtColor(cv2.imread(all_paths[file]),cv2.COLOR_BGR2GRAY)
    input_img = cv2.resize(input_img, (128,32))
    input_img = np.expand_dims(input_img, axis=2)
    input_img = input_img/255
    test_img.append(input_img)


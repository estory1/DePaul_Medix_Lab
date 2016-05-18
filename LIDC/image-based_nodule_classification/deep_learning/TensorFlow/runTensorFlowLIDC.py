#!/opt/local/bin/python2.7

# 20151110
# Evan Story (estory1@gmail.com)
#
# Implements Google's TensorFlow beginner's tutorial here: http://www.tensorflow.org/tutorials/mnist/pros/index.md

# import input_data   # src: https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/input_data.py

# To run with logging: ./runTensorFlowLIDC.py 2>&1 | tee output/`date +%Y%m%d-%H%M%S`-runTensorFlowLIDC.out.txt


# folder_path_to_mnist_data = "/Users/estory/Documents/syncable/School/DePaul/CSC578/final_proj/data/mnist/yann.lecun.com/exdb/mnist/"

# MNIST image data: if necessary, download first, then gunzip and read.
# mnist = input_data.read_data_sets(folder_path_to_mnist_data, one_hot=True)

import input_data_LIDC
import numpy as np
import math

import os
import pickle

import multiprocessing
import random


### Define hyperparmameters variables & functions.
batch_size = 10
nIters = 20000
validation_accuracy_min_cutoff = 0.99
learning_rate = 0.2



# # USE ONLY LIDC-IDRI-0001:
# dataset = input_data_LIDC.read_data_sets(
#   "../data/LIDC/resized_images-236x236/LIDC-IDRI-0001/",    # "../data/LIDC/resized_images/"
#   "*.tiff",
#   '/Users/estory/Documents/syncable/School/DePaul/research/LIDC_Complete_20141106/Extracts/master_join4.csv',
#   '/Users/estory/Documents/syncable/School/DePaul/research/LIDC_Complete_20141106/Extracts/DICOM_metadata_extracts/',
#   "imageSOP_UID-filePath-dicominfo-LIDC-IDRI-0001.csv")   # "*.csv"
# img_px_len_x = 236
# img_px_len_y = img_px_len_x
# X_len = img_px_len_x * img_px_len_y
# y_len = 5


# # # First 30 of the 236^2 crops.
# pickle_file_name = "Evans-MacBook-Pro.local-resized_images-236x236-first30.tensorflow.pickle"
# if os.path.isfile(pickle_file_name):
  # input_data_LIDC.esprint("Unpickling: " + pickle_file_name)
  # with open(pickle_file_name, "rb") as pickle_file:
  #   dataset_input = pickle.load(pickle_file)
# else:
#   dataset_input = input_data_LIDC.read_data_sets(
#     "../data/LIDC/resized_images-236x236-first30/",    # "../data/LIDC/resized_images/"  
#     "*.tiff",
#     '/Users/estory/Documents/syncable/School/DePaul/research/LIDC_Complete_20141106/Extracts/master_join4.csv',
#     '/Users/estory/Documents/syncable/School/DePaul/research/LIDC_Complete_20141106/Extracts/DICOM_metadata_extracts/',
#     "*.csv")   # "*.csv"
  # input_data_LIDC.esprint("Pickling: " + pickle_file_name)
  # with open(pickle_file_name, "wb") as pickle_file:
  #   pickle.dump(dataset_input, pickle_file)
# img_px_len_x = 236
# img_px_len_y = img_px_len_x
# X_len = img_px_len_x * img_px_len_y
# y_len = 5


# # # USE FULL DATASET.
# pickle_file_name = "Evans-MacBook-Pro.local-resized_images-236x236.tensorflow.pickle"
# if os.path.isfile(pickle_file_name):
#   input_data_LIDC.esprint("Unpickling: " + pickle_file_name)
#   with open(pickle_file_name, "rb") as pickle_file:
#     dataset_input = pickle.load(pickle_file)
# else:
#   dataset_input = input_data_LIDC.read_data_sets(
#     "../data/LIDC/resized_images-236x236/",
#     "*.tiff",
#     '../../../LIDC_Complete_20141106/Extracts/master_join4.csv',
#     '../../../LIDC_Complete_20141106/Extracts/DICOM_metadata_extracts/',
#     "*.csv")
#   input_data_LIDC.esprint("Pickling: " + pickle_file_name)
#   with open(pickle_file_name, "wb") as pickle_file:
#     pickle.dump(dataset_input, pickle_file)
# img_px_len_x = 236
# img_px_len_y = img_px_len_x
# X_len = img_px_len_x * img_px_len_y
# y_len = 5


# # USE FULL DATASET.
pickle_file_name = "Evans-MacBook-Pro.local-resized_images-32x32.tensorflow.pickle"
if os.path.isfile(pickle_file_name):
  input_data_LIDC.esprint("Unpickling: " + pickle_file_name)
  with open(pickle_file_name, "rb") as pickle_file:
    dataset_input = pickle.load(pickle_file)
else:
  dataset_input = input_data_LIDC.read_data_sets(
    "../data/LIDC/resized_images-32x32/",
    "*.tiff",
    '../../../LIDC_Complete_20141106/Extracts/master_join4.csv',
    '../../../LIDC_Complete_20141106/Extracts/DICOM_metadata_extracts/',
    "*.csv")
  input_data_LIDC.esprint("Pickling: " + pickle_file_name)
  with open(pickle_file_name, "wb") as pickle_file:
    pickle.dump(dataset_input, pickle_file)
img_px_len_x = 32
img_px_len_y = img_px_len_x
X_len = img_px_len_x * img_px_len_y
y_len = 5




# Randomize the image & label set in-place, taking care to maintain array correspondance.
# First, re-merge the training, validation, and test sets into a single set.
train_images, train_labels = dataset_input[0]
validation_images, validation_labels = dataset_input[1]
test_images, test_labels = dataset_input[2]

# 20160104: malignancy binning in [1,3] rather than [1,5]
def compare_arrays_with_order(a,b):
  # print a
  # print b
  for i,j in zip(a,b):
    if i != j:
      return False
  return True
def map_malignancy(malignancy_rating):
  if compare_arrays_with_order(malignancy_rating, [1,0,0,0,0]) or compare_arrays_with_order(malignancy_rating, [0,1,0,0,0]):
    return [1, 0, 0]
  elif compare_arrays_with_order(malignancy_rating, [0,0,1,0,0]):
    return [0, 1, 0]
  else:
    return [0, 0, 1]
train_labels = [map_malignancy(l) for l in train_labels]
validation_labels = [map_malignancy(l) for l in validation_labels]
test_labels = [map_malignancy(l) for l in test_labels]
y_len = 3


import itertools
images = list(itertools.chain(train_images, validation_images, test_images))
labels = list(itertools.chain(train_labels, validation_labels, test_labels))

combined = zip(images, labels)
random.shuffle(combined)
images[:], labels[:] = zip(*combined)

# Then, re-split the set.
n = np.shape(images)[0]

TRAIN_SIZE = int(math.floor(n * 0.7))
VALIDATION_SIZE = int(math.floor(TRAIN_SIZE * 0.1))

train_images = np.array(images[0:(TRAIN_SIZE - VALIDATION_SIZE)])
train_labels = np.array(labels[0:(TRAIN_SIZE - VALIDATION_SIZE)])

validation_images = np.array(images[(TRAIN_SIZE - VALIDATION_SIZE):TRAIN_SIZE])
validation_labels = np.array(labels[(TRAIN_SIZE - VALIDATION_SIZE):TRAIN_SIZE])

test_images = np.array(images[TRAIN_SIZE:])
test_labels = np.array(labels[TRAIN_SIZE:])


# Reconstitute data into DataSet objects, whose properties are referenced during training.
# train_images, train_labels = dataset_input[0]
# validation_images, validation_labels = dataset_input[1]
# test_images, test_labels = dataset_input[2]

print "train (raw): " + str(len(train_images)) + " : " + str(len(train_labels))
print "validation (raw): " + str(len(validation_images)) + " : " + str(len(validation_labels))
print "test (raw): " + str(len(test_images)) + " : " + str(len(test_labels))

dataset = input_data_LIDC.DataSets()
dataset.train = input_data_LIDC.DataSet(train_images, train_labels)
dataset.validation = input_data_LIDC.DataSet(validation_images, validation_labels)
dataset.test = input_data_LIDC.DataSet(test_images, test_labels)

print "train (DataSet): " + str(len(dataset.train._images)) + " : " + str(len(dataset.train._labels))
print "validation (DataSet): " + str(len(dataset.validation._images)) + " : " + str(len(dataset.validation._labels))
print "test (DataSet): " + str(len(dataset.test._images)) + " : " + str(len(dataset.test._labels))


# Check dataset for NaN values.
nan_in_images_train = np.any(np.isnan(dataset.train._images))
if nan_in_images_train:
  print("nan_in_images_train; data:")
  print(dataset.train._images)
nan_in_images_validation = np.any(np.isnan(dataset.validation._images))
if nan_in_images_validation:
  print("nan_in_images_validation; data:")
  print(dataset.validation._images)
nan_in_images_test = np.any(np.isnan(dataset.test._images))
if nan_in_images_test:
  print("nan_in_images_test; data:")
  print(dataset.test._images)

nan_in_labels_train = np.any(np.isnan(dataset.train._labels))
if nan_in_labels_train:
  print("nan_in_labels_train; data:")
  print(dataset.train._labels)
nan_in_labels_validation = np.any(np.isnan(dataset.validation._labels))
if nan_in_labels_validation:
  print("nan_in_labels_validation; data:")
  print(dataset.validation._labels)
nan_in_labels_test = np.any(np.isnan(dataset.test._labels))
if nan_in_labels_test:
  print("nan_in_labels_test; data:")
  print(dataset.test._labels)



import tensorflow as tf

# sess = tf.InteractiveSession()
num_cpu_cores = max(1, multiprocessing.cpu_count())
sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=num_cpu_cores,
                   intra_op_parallelism_threads=num_cpu_cores))

x = tf.placeholder("float", shape = [None, X_len])
y_ = tf.placeholder("float", shape = [None, y_len])

W = tf.Variable(tf.zeros([X_len, y_len]))
b = tf.Variable(tf.zeros([y_len]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)


# Function that introduces Gaussian-distributed noise into a variable.
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

# Function for max "pooling" over 2x2 blocks.
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2,2, 1], padding = "SAME")



### First convolutional layer.

# Weight convolution: computes 32 features over 1 input channel for each 5x5 patch.
W_conv1 = weight_variable([16,16, 1, 32])
# Bias vector with a component for each output channel.
b_conv1 = bias_variable([32])

# Reshape x to a 4D tensor. Dim 2 & 3 = width & height; dim 4 = # color channels.
x_image = tf.reshape(x, [-1, img_px_len_x, img_px_len_y, 1])

# Convolve x_image with the weight tensor, add bias, and apply a ReLU function...
conv1 = conv2d(x_image, W_conv1)
print("values in conv1:")
print(conv1)
# h_conv1 = tf.nn.relu(conv1 + b_conv1)
h_conv1 = tf.sigmoid(conv1 + b_conv1)
# ...and then apply the max pooling function over 2x2 blocks.
h_pool1 = max_pool_2x2(h_conv1)



### Second convolutional layer.

# Compute 64 features for each 5x5 patch.
W_conv2 = weight_variable([8,8, 32, 64])
b_conv2 = bias_variable([64])

# Convolve the first convolutional layer with the second conv. layer's weights, add bias, apply the ReLU function...
conv2 = conv2d(h_pool1, W_conv2)
print("values in conv2:")
print(conv2)

# h_conv2 = tf.nn.relu(conv2 + b_conv2)
h_conv2 = tf.sigmoid(conv2 + b_conv2)
# ...and then apply the max pooling function over 2x2 blocks.
h_pool2 = max_pool_2x2(h_conv2)



### Third convolutional layer.

# Compute 64 features for each 5x5 patch.
W_conv3 = weight_variable([4,4, 64, 64])
b_conv3 = bias_variable([64])

# Convolve the first convolutional layer with the second conv. layer's weights, add bias, apply the ReLU function...
conv3 = conv2d(h_pool2, W_conv3)
print("values in conv3:")
print(conv2)

# h_conv2 = tf.nn.relu(conv2 + b_conv2)
h_conv3 = tf.sigmoid(conv3 + b_conv3)
# ...and then apply the max pooling function over 2x2 blocks.
h_pool3 = max_pool_2x2(h_conv3)



### Densely connected layer.
#
# Image is reduced to a size of 7x7. Now we add a fully-connected layer with 1024 neurons,
# allowing processing on the whole image.
#
# The tensor is reshaped from the pooling layer into a batch of vectors, 
# and then the steps we executed in the conv. layers is performed again here.
reduced_img_len = (int(math.ceil(img_px_len_x/8)) * int(math.ceil(img_px_len_y/8)))
W_fc1 = weight_variable([ reduced_img_len * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, reduced_img_len * 64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1 = tf.sigmoid(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)



### Dropout. -- Reduces overfitting.
#
# Dropout is enabled during training, but disabled during testing.

# Create a placeholder for P(neuron retained during dropout).
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



### Readout layer -- where we add a softmax (i.e. logistic regression) layer.

# Setup weight matrix and bias vector again.
W_fc2 = weight_variable([1024, y_len])
b_fc2 = bias_variable([y_len])

# Run softmax over the dropout layer result for this layer's weights.
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



### Training and evaluation.
# Instead of SGD, here we'll use the "ADAM" optimizer. (Google says it's "more sophisticated".)
#
# Logging occurs every 100th step. Dropout rate is controlled by the keep_prob parameter in feed_dict.
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())

for i in range(nIters):
  train_batch = dataset.train.next_batch(batch_size)
  validation_batch = dataset.validation.next_batch(batch_size)

  if i % 100 == 0:
    train_accuracy = accuracy.eval(session=sess, feed_dict = { x: train_batch[0], y_: train_batch[1], keep_prob: 1.0 })
    input_data_LIDC.esprint("step %d, training accuracy %g" % (i, train_accuracy))

    validation_accuracy = accuracy.eval(session=sess, feed_dict = { x: dataset.validation.images, y_: dataset.validation.labels, keep_prob: 1.0 })
    input_data_LIDC.esprint("step %d, validation accuracy %g" % (i, validation_accuracy))
    if validation_accuracy >= validation_accuracy_min_cutoff:
      break

  train_step.run(session=sess, feed_dict = { x: train_batch[0], y_: train_batch[1], keep_prob: 0.5 })

# print "test accuracy %g" % accuracy.eval(feed_dict = { x: dataset.test.images, y_: dataset.test.labels, keep_prob: 1.0 })
input_data_LIDC.esprint("test accuracy %g" % accuracy.eval(session=sess, feed_dict = { x: dataset.test.images, y_: dataset.test.labels, keep_prob: 1.0 }))


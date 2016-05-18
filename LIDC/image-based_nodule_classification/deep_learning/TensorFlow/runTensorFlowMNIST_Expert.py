#!/opt/local/bin/python2.7

# 20151110
# Evan Story (estory1@gmail.com)
#
# Implements Google's TensorFlow beginner's tutorial here: http://www.tensorflow.org/tutorials/mnist/pros/index.md

import input_data   # src: https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/input_data.py
import numpy as np


folder_path_to_mnist_data = "/Users/estory/Documents/syncable/School/DePaul/CSC578/final_proj/data/mnist/yann.lecun.com/exdb/mnist/"

# MNIST image data: if necessary, download first, then gunzip and read.
mnist = input_data.read_data_sets(folder_path_to_mnist_data, one_hot=True)


import tensorflow as tf

sess = tf.InteractiveSession()

### Define variables & functions.
x = tf.placeholder("float", shape = [None, 784])
y_ = tf.placeholder("float", shape = [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


### Train the network.
for i in range(1000):
  batch = mnist.train.next_batch(50)
  # print(str(np.shape(batch[0])))
  # print(str(np.shape(batch[1])))
  # print(batch[0])
  # print(batch[1])
  train_step.run(feed_dict = { x: batch[0], y_: batch[1] })

### Evaluate performance.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

### Display result.
print accuracy.eval(feed_dict = { x: mnist.test.images, y_: mnist.test.labels })


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
W_conv1 = weight_variable([5, 5, 1, 32])
# Bias vector with a component for each output channel.
b_conv1 = bias_variable([32])

# Reshape x to a 4D tensor. Dim 2 & 3 = width & height; dim 4 = # color channels.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolve x_image with the weight tensor, add bias, and apply a ReLU function...
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# ...and then apply the max pooling function over 2x2 blocks.
h_pool1 = max_pool_2x2(h_conv1)



### Second convolutional layer.

# Compute 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# Convolve the first convolutional layer with the second conv. layer's weights, add bias, apply the ReLU function...
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# ...and then apply the max pooling function over 2x2 blocks.
h_pool2 = max_pool_2x2(h_conv2)



### Densely connected layer.
#
# Image is reduced to a size of 7x7. Now we add a fully-connected layer with 1024 neurons,
# allowing processing on the whole image.
#
# The tensor is reshaped from the pooling layer into a batch of vectors, 
# and then the steps we executed in the conv. layers is performed again here.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



### Dropout. -- Reduces overfitting.
#
# Dropout is enabled during training, but disabled during testing.

# Create a placeholder for P(neuron retained during dropout).
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



### Readout layer -- where we add a softmax (i.e. logistic regression) layer.

# Setup weight matrix and bias vector again.
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Run softmax over the dropout layer result for this layer's weights.
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



### Training and evaluation.
# Instead of SGD, here we'll use the "ADAM" optimizer. (Google says it's "more sophisticated".)
#
# Logging occurs every 100th step. Dropout rate is controlled by the keep_prob parameter in feed_dict.

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())

for i in range(20000):
  batch = mnist.train.next_batch(50)

  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict = { x:batch[0], y_: batch[1], keep_prob: 1.0 })
    print "step %d, training accuracy %g" % (i, train_accuracy)

  train_step.run(feed_dict = { x: batch[0], y_: batch[1], keep_prob: 0.5 })

print "test accuracy %g" % accuracy.eval(feed_dict = { x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0 })







### Define variables, session.
# x = tf.placeholder("float", [None, 784])  # input vector

# W = tf.Variable(tf.zeros([784,10]))       # weight matrix
# b = tf.Variable(tf.zeros([10]))           # bias vector

# # logistic regression applied to [x * W] + b
# y = tf.nn.softmax(tf.matmul(x,W) + b)

# y_ = tf.placeholder("float", [None,10])

# # Compute error function (cost function).
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# # Run stochastic gradient descent (SGD) over the error surface.
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# init = tf.initialize_all_variables()

# # Run a TF Session, i.e. a mathematical universe containing the variables and behaviors defined above.
# sess = tf.Session()
# sess.run(init)



# ### Train the network.
# for i in range(1000):
#   batch_xs, batch_ys = mnist.train.next_batch(100)
#   sess.run(train_step, feed_dict = { x: batch_xs, y_: batch_ys })


# ### Evaluate our network.
# # Compute the ratio of predicted to actual outputs.
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# print sess.run(accuracy, feed_dict = { x: mnist.test.images, y_: mnist.test.labels })
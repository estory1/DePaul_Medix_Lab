#!/opt/local/bin/python2.7

# 20151110
# Evan Story (estory1@gmail.com)
#
# Implements Google's TensorFlow beginner's tutorial here: http://www.tensorflow.org/tutorials/mnist/beginners/index.md

import input_data   # src: https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/input_data.py


folder_path_to_mnist_data = "/Users/estory/Documents/syncable/School/DePaul/CSC578/final_proj/data/mnist/yann.lecun.com/exdb/mnist/"

# MNIST image data: if necessary, download first, then gunzip and read.
mnist = input_data.read_data_sets(folder_path_to_mnist_data, one_hot=True)


import tensorflow as tf

### Define variables, session.
x = tf.placeholder("float", [None, 784])  # input vector

W = tf.Variable(tf.zeros([784,10]))       # weight matrix
b = tf.Variable(tf.zeros([10]))           # bias vector

# logistic regression applied to [x * W] + b
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])

# Compute error function (cost function).
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Run stochastic gradient descent (SGD) over the error surface.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

# Run a TF Session, i.e. a mathematical universe containing the variables and behaviors defined above.
sess = tf.Session()
sess.run(init)



### Train the network.
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict = { x: batch_xs, y_: batch_ys })


### Evaluate our network.
# Compute the ratio of predicted to actual outputs.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict = { x: mnist.test.images, y_: mnist.test.labels })
#!/opt/local/bin/python2.7
#
# Tests whether your TensorFlow library will execute on your GPU. Output interpretation:
#
# Extends: https://www.tensorflow.org/versions/r0.9/how_tos/using_gpu/index.html

import unittest
import tensorflow as tf

## Creates a graph.
#a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#c = tf.matmul(a, b)
## Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
## Runs the op.
#print sess.run(c)


def exec_on_GPU():
  # Creates a graph.
  with tf.device('/gpu:0'):                 # This line forces TF to look for a GPU. If none, then an error is thrown.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
  # Creates a session with log_device_placement set to True.
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
  # Runs the op.
  print sess.run(c)


class TestGPU(unittest.TestCase):
  def test(self):
    try:
      exec_on_GPU()
      print "\nYES, TensorFlow will execute on your GPU."
    except tf.python.framework.errors.InvalidArgumentError as e:
      print "\nNO, TensorFlow will not execute on your GPU."

if __name__ == '__main__':
  unittest.main()
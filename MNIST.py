from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def CNN(X):
  
  inImg = tf.reshape(X, [-1, 28, 28, 1]) #inout image reshaping

  #1st convolutional layer-image -> 32 features
  W1 = weights([5, 5, 1, 32])
  B1 = biases([32])
  H1 = tf.nn.relu(conv2d(inImg, W1) + B1)
  Hpool1 = pool2x(H1) # downsample by 2x

  

  # Second convolutional layer 32 feature -> 64.
  W2 = weights([5, 5, 32, 64])
  B2 = biases([64])
  H2 = tf.nn.relu(conv2d(Hpool1, W2) + B2)
  Hpool2 = pool2x(H2) # pooling again
  
    

  # Fully connected layer 1,  7x7x64 feature from downsampling to 1024 features.
  Wfc1 = weights([7 * 7 * 64, 1024])
  Bfc1 = biases([1024])

  flattenPool2 = tf.reshape(Hpool2, [-1, 7*7*64])
  Hfc1 = tf.nn.relu(tf.matmul(flattenPool2, Wfc1) + Bfc1)

  # Dropout to reduce overfitting
  ProbaKeepng = tf.placeholder(tf.float32)
  Hfc1Drop = tf.nn.dropout(Hfc1, ProbaKeepng)

  # 1024 features -> 10 classes, one for each digit
  
  Wfc2 = weights([1024, 10])
  Bfc2 = biases([10])

  outConv = tf.matmul(Hfc1Drop, Wfc2) + Bfc2
  return outConv, ProbaKeepng

def pool2x(X):
  return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv2d(X, W):
  return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')


def weights(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def biases(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):

  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  X = tf.placeholder(tf.float32, [None, 784])

  Y = tf.placeholder(tf.float32, [None, 10])

  outConv, ProbaKeepng = CNN(X)

  XEntropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=outConv)
  XEntropy = tf.reduce_mean(XEntropy)

  step = tf.train.AdamOptimizer(1e-4).minimize(XEntropy)

  #with tf.name_scope('accuracy'):
  correctPred = tf.equal(tf.argmax(outConv, 1), tf.argmax(Y, 1))
  correctPred = tf.cast(correctPred, tf.float32)
  accuracy = tf.reduce_mean(correctPred)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1], ProbaKeepng: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      
      step.run(feed_dict={X: batch[0], Y: batch[1], ProbaKeepng: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels, ProbaKeepng: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
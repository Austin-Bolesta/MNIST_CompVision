import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

#init weights and biases 
weights = tf.Variable(tf.zeros([784,10]))
bias = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

outMul = tf.matmul(X, weights) + bias

#use cross entropy to optimize
crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=outMul))

step = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropy)

for _ in range(1000):
  batchTrain = mnist.train.next_batch(100)
  step.run(feed_dict={X: batchTrain[0], Y: batchTrain[1]})

correctPred = tf.equal(tf.argmax(outMul,1), tf.argmax(Y,1))
accr = tf.reduce_mean(tf.cast(correctPred, tf.float32))
print(accr.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

# 1st Convolutional Layer 
WConv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
BConv1 = tf.Variable(tf.constant(0.1, shape=[32]))
xImg = tf.reshape(X, [-1,28,28,1])
hConv1 = tf.nn.relu(tf.nn.conv2d(xImg, WConv1, strides=[1, 1, 1, 1], padding='SAME') + BConv1)
hPool1 = tf.nn.max_pool(hConv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 2nd Convolutional Layer 
WConv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
BConv2 = tf.Variable(tf.constant(0.1, shape=[64]))

hConv2 = tf.nn.relu(tf.nn.conv2d(hPool1, WConv2, strides=[1, 1, 1, 1], padding='SAME') + BConv2)
hPool2 = tf.nn.max_pool(hConv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#fully connected layer
WfullyCon1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
BfullyCon1 = tf.Variable(tf.constant(0.1, shape=[1024]))

hPool2Flat = tf.reshape(hPool2, [-1, 7*7*64])
hfullyCon1 = tf.nn.relu(tf.matmul(hPool2Flat, WfullyCon1) + BfullyCon1)

# neuron dropout to reduce overfitting
keepProba = tf.placeholder(tf.float32)
fullDropout = tf.nn.dropout(hfullyCon1, keepProba)


WfullyCon2 = tf.truncated_normal([1024, 10], stddev=0.1)
BfullyCon2 = tf.Variable(tf.constant(0.1, shape=[10]))

YConv = tf.matmul(fullDropout, WfullyCon2) + BfullyCon2

crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=YConv))
step = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)
correctPred = tf.equal(tf.argmax(YConv, 1), tf.argmax(Y, 1))
accr = tf.reduce_mean(tf.cast(correctPred, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i % 100 == 0:
			trainAccr = accr.eval(feed_dict={X: batch[0], Y: batch[1], keepProba: 1.0})
			print('step {0}, training accuracy {1:0.3f}'.format(i, trainAccr))
		step.run(feed_dict={X: batch[0], Y: batch[1], keepProba: 0.5})
	print('test accuracy {1:0.3f}'.format(accr.eval(feed_dict={
		X: mnist.test.images, Y: mnist.test.labels, keepProba: 1.0})))


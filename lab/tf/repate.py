#!coding:utf-8

import numpy as np
import tensorflow as tf

BATCH_SIZE = 8
seed = 13455

rng = np.random.RandomState()
X = rng.rand(32, 2)
Y = [[int(i+j)] for (i, j) in X]

x = tf.placeholder(tf.float32, shape=(None, 2))
y_= tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], mean=0, stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], mean=0, stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print "w1:\n", w1
	print "w2:\n", w2
	print"\n"
	
	STEPS = 3000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		if i%500==0:
			total_loss = sess.run(loss, feed_dict={x:X, y_:Y})
			print total_loss
	print "\n"
	print "w1:\n", sess.run(w1)
	print "w2:\n", sess.run(w2)
			

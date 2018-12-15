#!coding:utf-8

import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
PROFIT = 1
COST = 9

#基于seed产生随机数
rmd = np.random.RandomState(SEED)
#随机数返回32行2列的矩阵 表示32组体积和重量 作为输入数据集
X = rmd.rand(32, 2)
#从X这个32行2列的矩阵中取出一行判断如果和小于1给Y赋值1 反之给Y赋值0
#作为输入数据集的标签（正确答案）
Y = [[x0+x1+(rmd.rand()/10.0-0.05)] for (x0, x1) in X]
#print "X:\n", X
#print "Y:\n", Y

#1定义神经网络的输入，参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_= tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, mean=0, seed=1))

y = tf.matmul(x, w1)

#2定义损失函数及反向传播方法
#loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_mse = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*COST, (y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
#train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss_mse)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)

#3生成会话，训练STEPS轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	#训练模型
	STEPS = 200000000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x: X[start:end], y_:Y[start:end]})
		if i%500==0:
			print "After %d training steps, w1 is:" % i
			print sess.run(w1)
	print "Fainal w1 is : \n", sess.run(w1)


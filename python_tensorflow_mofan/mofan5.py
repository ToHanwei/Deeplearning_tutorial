# -*- coding: utf-8 -*-
#第5课：例子2练习

import tensorflow as tf
import numpy as np

x=np.random.rand(100)
y=x*0.1+0.3


Weight=tf.Variable(tf.random_uniform([1],-1,1))
baises=tf.Variable(tf.zeros([1]))

y_pd=Weight*x+baises


loss=tf.reduce_mean(tf.square(y-y_pd))
gdo=tf.train.GradientDescentOptimizer(0.5)
train=gdo.minimize(loss)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

for i in range(201):
    sess.run(train)
    if i%20==0:
        print sess.run([Weight,baises])


# -*- coding: utf-8 -*-
#第11课：创建神经网络

import tensorflow as tf
import numpy as np

def add_layer(inputs,input_size,output_size,activation_function=None):
    Weight=tf.Variable(tf.random_normal((input_size,output_size)))
    Biase=tf.Variable(tf.zeros((1,output_size))+0.01)
    wpb=tf.matmul(inputs,Weight)+Biase

    if activation_function is None:
        ret=wpb
    else:
        ret=activation_function(wpb)
    return ret

x_data=np.linspace(-1,1,30)[:,np.newaxis]
print np.linspace(-1,1,30)
print x_data
print x_data.shape
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

x_ph=tf.placeholder(tf.float32,[None,1])
y_ph=tf.placeholder(tf.float32,[None,1])

l1=add_layer(x_ph,1,10,activation_function=tf.nn.relu)
output=add_layer(l1,10,1,activation_function=None)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(y_data-output),reduction_indices=[1]))
gdo=tf.train.GradientDescentOptimizer(0.1)
train=gdo.minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(0,1000):
        sess.run(train,feed_dict={x_ph:x_data,y_ph:y_data})
        if i%50==0:
            print sess.run(loss,feed_dict={x_ph:x_data,y_ph:y_data})


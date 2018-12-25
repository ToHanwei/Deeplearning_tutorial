# -*- coding: utf-8 -*-
#第16课：分类学习

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,input_size,output_size,activation_function=None):
    Weight=tf.Variable(tf.random_normal((input_size,output_size)))
    Biase=tf.Variable(tf.zeros((1,output_size))+0.01)
    wpb=tf.matmul(inputs,Weight)+Biase

    if activation_function is None:
        ret=wpb
    else:
        ret=activation_function(wpb)
    return ret
def compare_result(x_data,y_lables):
    global prediction
    y_predict=sess.run(prediction,feed_dict={x_ph:x_data})
    label_equal=tf.equal(tf.argmax(y_predict,1),tf.argmax(y_lables,1))
    accu=tf.reduce_mean(tf.cast(label_equal,tf.float32))
    result =sess.run(accu)
    return result


x_ph=tf.placeholder(tf.float32,[None,784])
y_ph=tf.placeholder(tf.float32,[None,10])
prediction=add_layer(x_ph,784,10,activation_function=tf.nn.softmax)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ph*tf.log(prediction),reduction_indices=[1]))

loss=cross_entropy
gdo=tf.train.GradientDescentOptimizer(0.5)
train=gdo.minimize(loss)

init=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        x_batch,y_batch=mnist.train.next_batch(100)
        sess.run(train,feed_dict={x_ph:x_batch,y_ph:y_batch})
        if i%50==0:
            # print sess.run(loss,feed_dict={x_ph:x_batch,y_ph:y_batch})
            print compare_result(mnist.test.images,mnist.test.labels)

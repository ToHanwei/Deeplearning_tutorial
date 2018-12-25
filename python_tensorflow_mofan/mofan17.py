# -*- coding: utf-8 -*-
#第17课：overfitting

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

digit=load_digits()
x_data=digit.data
y_label=digit.target
y_label=LabelBinarizer().fit_transform(y_label)
x_train,x_test,y_train,y_test=train_test_split(x_data,y_label,test_size=0.3)


def add_layer(inputs,input_size,output_size,n_layer,activation_function=None):
    layer_name="layer%s"%n_layer

    Weight=tf.Variable(tf.random_normal((input_size,output_size)))
    Weight=tf.nn.dropout(Weight,keep_prob=keep_ph)
    Biase=tf.Variable(tf.zeros((1,output_size))+0.01)
    wpb=tf.matmul(inputs,Weight)+Biase

    if activation_function is None:
        ret=wpb
    else:
        ret=activation_function(wpb)
    tf.histogram_summary(layer_name+'/outputs',ret)
    return ret
def compare_result(x_data,y_lables):
    global prediction
    y_predict=sess.run(prediction,feed_dict={x_ph:x_data})
    label_equal=tf.equal(tf.argmax(y_predict,1),tf.argmax(y_lables,1))
    accu=tf.reduce_mean(tf.cast(label_equal,tf.float32))
    result =sess.run(accu)
    return result


x_ph=tf.placeholder(tf.float32,[None,64])
y_ph=tf.placeholder(tf.float32,[None,10])
keep_ph=tf.placeholder(tf.float32)
l1=add_layer(x_ph,64,50,n_layer=1,activation_function=tf.nn.tanh)
prediction=add_layer(l1,50,10,n_layer=2,activation_function=tf.nn.softmax)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ph*tf.log(prediction),reduction_indices=[1]))

loss=cross_entropy
tf.scalar_summary('loss',loss)
gdo=tf.train.GradientDescentOptimizer(0.6)
train=gdo.minimize(loss)

merge=tf.merge_all_summaries()
init=tf.initialize_all_variables()
with tf.Session() as sess:
    train_writer=tf.train.SummaryWriter('logs17/train/',sess.graph)
    test_wirter=tf.train.SummaryWriter('logs17/test/',sess.graph)
    sess.run(init)
    for i in range(500):
        sess.run(train,feed_dict={x_ph:x_train,y_ph:y_train,keep_ph:0.5})
        if i%50==0:
            train_merge=sess.run(merge,feed_dict={x_ph:x_train,y_ph:y_train,keep_ph:1})
            test_merge=sess.run(merge,feed_dict={x_ph:x_test,y_ph:y_test,keep_ph:1})
            train_writer.add_summary(train_merge,i)
            test_wirter.add_summary(test_merge,i)






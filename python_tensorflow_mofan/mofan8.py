# -*- coding: utf-8 -*-
#第8课：placeholder

import tensorflow as tf

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.mul(input1,input2)

with tf.Session() as sess:
    print sess.run(output,feed_dict={input1:[0.7],input2:[0.2]})
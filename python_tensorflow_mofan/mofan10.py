# -*- coding: utf-8 -*-
#第10课：添加层

import tensorflow as tf


def add_layer(inputs,input_size,output_size,activation_function=None):
    Weight=tf.Variable(tf.random_normal((input_size,output_size)))
    Biase=tf.Variable(tf.zeros((1,output_size))+0.01)
    wpb=tf.matmul(inputs,Weight)+Biase

    if activation_function is None:
        ret=wpb
    else:
        ret=activation_function(wpb)
    return ret

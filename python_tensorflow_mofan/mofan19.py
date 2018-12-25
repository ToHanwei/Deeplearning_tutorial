# -*- coding: utf-8 -*-
#第19课：模型保存

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Weight=tf.Variable([[1,2,3],[4,5,6]],dtype=tf.float32)
# Baise=tf.Variable([[1,2,3]],dtype=tf.float32)
#
# init=tf.initialize_all_variables()
# saver=tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     saver.save(sess,"save/mofan19.ckpt")


Weight=tf.Variable(np.arange(6).reshape(2,3),dtype=tf.float32)
Baise=tf.Variable(np.arange(3).reshape(1,3),dtype=tf.float32)
saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'save/mofan19.ckpt')
    print sess.run(Weight),sess.run(Baise)
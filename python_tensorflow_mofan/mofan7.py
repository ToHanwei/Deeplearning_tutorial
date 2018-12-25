# -*- coding: utf-8 -*-
#第7课：变量

import tensorflow as tf

state=tf.Variable(0.1,dtype=tf.float32,name='counter')
# print state.name
one=tf.constant(1,dtype=tf.float32)

new_value=tf.add(state,one)
update=tf.assign(state,new_value)

init=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        print sess.run(update)
        print sess.run(state)
    # print sess.run(tf.random_uniform([1],-1,1))
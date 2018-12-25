# -*- coding: utf-8 -*-
#第6课：session
import tensorflow as tf

mat1=tf.constant([[3,3]])
mat2=tf.constant([[2],
                 [2]])
product=tf.matmul(mat1,mat2)

# sess=tf.Session()
# print sess.run(product)
# sess.close()

with tf.Session() as sess:
    print sess.run(product)
import tensorflow as tf

v1 = tf.constant([[2, 3]])
v2 = tf.constant([[2], [3]])

product = tf.matmul(v1, v2)
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

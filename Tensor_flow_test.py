import tensorflow as tf

with tf.Session():
    x = tf.constant([[5,6],[7,8]])
    print(tf.shape(x))
    print(tf.rank(x))
    z = tf.matmul(x,x)
    result = z.eval()
    print(result)

# x = tf.constant("Hello Tensorflow")
# print(x)
# sess = tf.Session()
#
# print(sess.run(x))
import tensorflow as tf 


x = tf.placeholder(tf.float32, shape = [1, 1])
m = tf.matmul(x, x)

with tf.Session() as sess: 
	m_out = sess.run(m, feed_dict = {x:[[2.0]]})

print(m_out)

import tensorflow.contrib.eager as tfe 
tfe.enable_eager_execution()

a = tf.constant(12)
counter = 0
while not tf.equal(a, 1):
	if tf.equal(a % 2, 0):
		a = a / 2
	else:
		a = 3 * a + 1
	print(a)


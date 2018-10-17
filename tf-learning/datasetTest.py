import tensorflow as tf 
import numpy as np 

import time

dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

stime = time.time()

with tf.Session() as sess:
	for i in range(5):
		ctime = time.time()
		print("%f second." % (ctime - stime))
		stime = ctime
		print(sess.run(one_element))


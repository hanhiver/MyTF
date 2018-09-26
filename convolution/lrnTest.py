import tensorflow as tf 
import numpy as np 

input_data = np.array([i for i in range(0, 32)])
reshape_data = tf.reshape(input_data, [2, 2, 2, 4])
reshape_data_float = tf.to_float(reshape_data)

lrn = tf.nn.local_response_normalization(
	input = reshape_data_float, 
	depth_radius = 2,
	bias = 0, 
	alpha = 1,
	beta = 1)

with tf.Session() as sess: 
	print('---------------- Before LRN ------------------')
	print(reshape_data_float.eval())
	print('---------------- After  LRN ------------------')
	print(lrn.eval())


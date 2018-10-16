import tensorflow as tf 
import tensorflow.contrib.eager as tfe 
import os

import numpy as np 

tfe.enable_eager_execution()

#dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
#dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size = (5, 2)))
dataset = tf.data.Dataset.from_tensor_slices(
	{
		"a" : np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 
		"b" : np.random.uniform(size = (5, 2))
	}
)

for one_element in tfe.Iterator(dataset):
	print(one_element)
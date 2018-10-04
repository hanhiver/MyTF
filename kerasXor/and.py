import os 
import tensorflow as tf  
from tensorflow import keras

import numpy as np 

# input data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_labels = np.array([0, 0, 0, 1])

model = keras.Sequential()
model.add(keras.layers.Dense(1, activation = tf.nn.sigmoid, input_dim = 2))

optimizer = keras.optimizers.SGD(lr = 0.05, decay = 1e-6, momentum = 0.9, nesterov = True)

model.compile(
	loss = 'binary_crossentropy', 
	optimizer = optimizer,
	metrics = ['accuracy']
	)

history = model.fit(
	x_train, 
	y_labels,
	batch_size = 1, 
	epochs = 100, 
	shuffle = True,
	verbose = 1, 
	validation_split = 0.0
	)

loss_metrics = model.evaluate(x_train, y_labels, batch_size = 1)
print(loss_metrics)

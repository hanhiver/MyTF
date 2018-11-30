import os

import tensorflow as tf 

import random
import numpy as np 

# Set the length of BP sequence.
truncated_backprop_length = 10
state_size = 10
batch_size = 10

def generateData(n):
	xs = []
	ys = []

	for i in range(2000):
		# Genrate next number, from 1 to 50.
		k = random.uniform(1, 50)

		x_seq = [np.sin(k + j) for j in range(0, n)]
		y_seq = [np.sin(k + n)]

		xs.append(x_seq)
		ys.append(y_seq)

	train_x = np.array(xs[0:1500])
	train_y = np.array(ys[0:1500])
	test_x = np.array(xs[1500:])
	test_y = np.array(ys[1500:])

	return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = generateData(truncated_backprop_length)

# Set the placeholder for input x and output y. 
X_placeholder = tf.placeholder(tf.float32, [None, truncated_backprop_length])
Y_placeholder = tf.placeholder(tf.float32, [None, 1])

# Set a matrix to store the status. 
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# Set the weights and biases
W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype = tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype = tf.float32)
W2 = tf.Variable(np.random.rand(state_size, 1), dtype = tf.float32)
b2 = tf.Variable(np.zeros(1), dtype = tf.float32)

# input_series = [1, 2, 3, ... 2000]. 
# New input_series = [[1, 2, ... 10], [11, 12... 20] ... [ ... 2000]]
input_series = tf.unstack(X_placeholder, axis = 1)

# current states = [ [1, 2, ... 10], ... [ 91, 92 ... 100]]
current_state = init_state

for current_input in input_series:
	# current_input = [1, 2, 3... 10]
	# New current_input = [[1], [2], ... [10]]
	current_input = tf.reshape(current_input, [batch_size, 1])

	# input_and_state_concatenated = [[1, s1, s2... s10], [2, s1, s2, ... s10], [10, s1, s2, ... s10]]
	input_and_state_concatenated = tf.concat([current_input, current_state], 1)

	next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)
	current_state = next_state

# Define the loss function
logits = tf.matmul(current_state, W2) + b2 
loss = tf.square(tf.subtract(Y_placeholder, logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Set the GPU environment. 
# Use 50% gpu memory. 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto(log_device_placement = False)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:
	sess.run(tf.global_variables_initializer())
	_current_state = np.zeros((batch_size, state_size))

	for epoch_id in range(101):
		for batch_id in range(len(train_x) // batch_size):
			begin = batch_id * batch_size
			end = begin + batch_size

			# Get a batch data from the training data. 
			batchX = train_x[begin:end]
			batchY = train_y[begin:end]

			_train_step, _current_state = sess.run(
				[train_step, current_state], 
				feed_dict = {
				X_placeholder:batchX, 
				Y_placeholder:batchY,
				init_state:_current_state
				})

			test_indices = np.arange(len(test_x))
			np.random.shuffle(test_indices)
			test_indices = test_indices[0:10]
			x = test_x[test_indices]
			y = test_y[test_indices]

			val_loss = np.mean(sess.run(
				loss,
				feed_dict = {
				X_placeholder:x,
				Y_placeholder:y,
				init_state:_current_state
				}))
		if epoch_id % 10 == 0:
			print('epoch: {}, loss: {}'.format(epoch_id, val_loss))

	# Write the session to graph.
	writer = tf.summary.FileWriter('./my_graph/1', sess.graph)
	# writer.add_graph(sess.graph)

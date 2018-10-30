""" 
CapsNet-mnist.py
Function: Tensorflow实现胶囊神经网络识别MNIST数据集。
Dong Han 2018.10.30
"""
import os 

import numpy as np 
import tensorflow as tf 

import input_data 

mnist = input_data.read_data_sets("../MNIST/MNIST_data/")

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

""" 输入图像 """
X = tf.placeholder(
	shape = [None, 28, 28, 1], 
	dtype = tf.float32, 
	name = "X")

""" 卷积层 """
conv1_params = {
	"filters": 256, 
	"kernel_size": 9,
	"strides": 1,
	"padding": "valid",
	"activation": tf.nn.relu
}

conv1 = tf.layers.conv2d(X, name = 'conv1', **conv1_params)

""" 主胶囊层 """
caps1_num = 32
caps1_dims = 8

conv2_params = {
	"filters": (caps1_num * caps1_dims), # 256卷积filters
	"kernel_size": 9, 
	"strides": 2, 
	"padding": "valid", 
	"activation": tf.nn.relu 
}

conv2 = tf.layers.conv2d(conv1, name = 'conv2', **conv2_params)

caps1_caps = caps1_num * 6 * 6

caps1_raw = tf.reshape(conv2, [-1, caps1_caps, caps1_dims], name = 'caps1_raw')

def squash(s, axis = -1, epsilon = 1e-7): 
	s_sqr_norm = tf.reduce_sum(tf.square(s), axis = axis, keepdims = True)
	V = s_sqr_norm / (1.0 + s_sqr_norm) / tf.sqrt(s_sqr_norm + epsilon)

	return V * s 

caps1_output = squash(caps1_raw)

""" 数字胶囊层 """
caps2_caps = 10
caps2_dims = 16

routing_num = 2

init_sigma = 0.01 

W_init = tf.random_normal(
	shape = (1, caps1_caps, caps2_caps, caps2_dims, caps1_dims), 
	stddev = init_sigma, 
	dtype = tf.float32, 
	name = 'W_init')

W = tf.Variable(W_init, name = 'W')

batch_size = tf.shape(X)[0]

W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name = "W_tiled")

# 数组 u
caps1_output_expanded = tf.expand_dims(
	caps1_output, 
	-1, 
	name = 'caps1_output_expanded')

caps1_output_tile = tf.expand_dims(
	caps1_output_expanded, 
	2, 
	name = 'caps1_output_tile')

caps1_output_tiled = tf.tile(
	caps1_output_tile, 
	[1, 1, caps2_caps, 1, 1], 
	name = 'caps1_output_tiled')

# 数组 u_hat
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name = 'caps2_predicted')

# 路由算法
raw_weights = tf.zeros(
	[batch_size, caps1_caps, caps2_caps, 1, 1], 
	dtype = tf.float32, 
	name = 'raw_weights')

b = raw_weights 

for i in range(routing_num): 
	c = tf.nn.softmax(b, axis = 2) # b == raw weights
	preds = tf.multiply(c, caps2_predicted)
	s = tf.reduce_sum(preds, axis = 1, keepdims = True)
	vj = squash(s, axis = -2)

	if i < routing_num -1: 
		vj_tiled = tf.tile(vj, [1, caps1_caps, 1, 1, 1], name = 'vj_tiled')
		agreement = tf.matmul(
			caps2_predicted, 
			vj_tiled, 
			transpose_a = True, 
			name = 'agreement')

		b += agreement

caps2_output = vj 

# 求预测向量长度
def safe_norm(s, axis = -1, epsilon = 1e-7, keep_dims = False, name = None):
	with tf.name_scope(name, default_name = 'safe_norm'):
		squared_norm = tf.reduce_sum(tf.square(s), axis = axis, 
									keep_dims = keep_dims)
		return tf.sqrt(squared_norm + epsilon)

y_proba = safe_norm(vj, axis = -2, name = 'y_proba')
y_proba_argmax = tf.argmax(y_proba, axis = 2, name = 'y_proba')
y_pred = tf.squeeze(y_proba_argmax, axis = [1, 2], name = 'y_pred')

# 标签
y = tf.placeholder(shape = [None], dtype = tf.int64)

# 边际损失
m_plus = 0.9 
m_minus = 0.1 
lambda_ = 0.5 

T = tf.one_hot(y, depth = caps2_caps, name = 'T')

caps2_output_norm = safe_norm(
	caps2_output, 
	axis = -2, 
	keep_dims = True, 
	name = 'caps2_output_norm')

present_error_raw = tf.square(
	tf.maximum(0.0, m_plus - caps2_output_norm),
	name = 'present_error_raw')

present_error = tf.reshape(
	present_error_raw, 
	shape = (-1, 10), 
	name = 'present_error')

absent_error_raw = tf.square(
	tf.maximum(0.0, caps2_output_norm - m_minus),
	name = 'absent_error_raw')

absent_error = tf.reshape(
	absent_error_raw, 
	shape = (-1, 10), 
	name = 'absent_error')

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name = 'L')

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis = 1), name = 'margin_loss')

# Mask
mask_with_labels = tf.placeholder_with_default(
	False, 
	shape = (), 
	name = 'mask_with_labels')

reconstruction_targets = tf.cond(
	mask_with_labels,  # condition
	lambda: y,         # if True
	lambda: y_pred,    # if False
	name = 'reconstruction_targets')

reconstruction_mask = tf.one_hot(
	reconstruction_targets, 
	depth = caps2_caps,
	name = 'reconstruction_mask')

reconstruction_mask_reshape = tf.reshape(
	reconstruction_mask, 
	[-1, 1, caps2_caps, 1, 1], 
	name = 'reconstruction_mask_reshape')

caps2_output_masked = tf.multiply(
	caps2_output, 
	reconstruction_mask_reshape, 
	name = 'caps2_output_masked')

# 解码器
n_hidden1 = 512
n_hidden2 = 1024
n_output = 28 * 28

decoder_input = tf.reshape(
	caps2_output_masked, 
	[-1, caps2_caps * caps2_dims], 
	name = 'decoder_input')

with tf.name_scope("decoder"):
	hidden1 = tf.layers.dense(
		decoder_input, 
		n_hidden1, 
		activation = tf.nn.relu,
		name = 'hidden1')

	hidden2 = tf.layers.dense(
		hidden1, 
		n_hidden2,
		activation = tf.nn.relu, 
		name = 'hidden2')

	decoder_output = tf.layers.dense(
		hidden2, 
		n_output, 
		activation = tf.nn.sigmoid,
		name = 'decoder_output')

# 重构损失
X_flat = tf.reshape(X, [-1, n_output], name = 'X_flat')

squared_difference = tf.square(X_flat - decoder_output, name = 'squared_difference')
reconstruction_loss = tf.reduce_sum(squared_difference, name = 'reconstruction_loss')

# 最终损失
alpha = 0.0005
loss = tf.add(margin_loss, alpha * reconstruction_loss, name = 'loss')

# 额外的设置

# 准确度
correct = tf.equal(y, y_pred, name = 'correct')
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = 'accuracy')

# 使用Adam优化器
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name = 'training_op')

# 全局初始化
init = tf.global_variables_initializer()
saver = tf.train.Saver()

####################################

# 训练模型
n_epochs = 3
batch_size = 50
restore_checkpoint = False

n_iterations_per_epoch = mnist.train.num_examples // batch_size
n_iterations_validation = mnist.validation.num_examples // batch_size

print("-----", n_iterations_validation)
best_loss_val = np.infty
checkpoint_path = '/tmp/capsNetMnist/my_capsule_network'


with tf.Session() as sess:
	if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
		saver.restore(sess, checkpoint_path)
	else:
		init.run()

	print(" Train the model with training data. ")
	print("=====================================\n")

	for epoch in range(n_epochs):
		for iteration in range(1, n_iterations_per_epoch + 1):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			# 运行训练操作并计算损失
			_, loss_train = sess.run(
				[training_op, loss],
				feed_dict = { X: X_batch.reshape([-1, 28, 28, 1]),
							  y: y_batch, 
							  mask_with_labels: True })
			print("\rIteration: {}/{} ({:.1f}%) ---  Loss: {:.5f}".format(
				iteration,
				n_iterations_per_epoch,
				iteration * 100 / n_iterations_per_epoch,
				loss_train), end = "")

		# 在每一个epoch结束后，用validation数据检测损失和准确度。
		loss_vals = []
		acc_vals = []

		for iteration in range(1, n_iterations_validation + 1):
			X_batch, y_batch = mnist.validation.next_batch(batch_size)
			loss_val, acc_val = sess.run(
				[loss, accuracy], 
				feed_dict = { X: X_batch.reshape([-1, 28, 28, 1]), 
							  y: y_batch})

			loss_vals.append(loss_val)
			acc_vals.append(acc_val)

			print("\rIteration: {}/{} ({:.1f}%) ---  Loss: {:.5f}".format(
				iteration,
				n_iterations_per_epoch,
				iteration * 100 / n_iterations_per_epoch,
				loss_train), end = "") 

		loss_val = np.mean(loss_vals)
		acc_val = np.mean(acc_vals)
		print("\nEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
			epoch + 1, acc_val * 100, loss_val,
			" (improved)\n" if loss_val < best_loss_val else "(not improved)\n"))

		# 如果准确度有进步，保存模型。
		if loss_val < best_loss_val: 
			save_path = saver.save(sess, checkpoint_path)
			best_loss_val = loss_val

# 用mnist中的测试数据检验模型
n_iterations_test = mnist.test.num_examples // batch_size

with tf.Session() as sess: 
	saver.restore(sess, checkpoint_path)

	loss_tests = []
	acc_tests = []

	print(" Evaluating the model with test data. ")
	pritn("=====================================\n")

	for iteration in range(1, n_iterations_test + 1):
		X_batch, y_batch = mnist.test.next_batch(batch_size)
		loss_test, acc_test = sess.run(
				[loss, accuracy], 
				feed_dict = { X: X_batch.reshape([-1, 28, 28, 1]), 
							  y: y_batch})
		loss_tests.append(loss_test)
		acc_tests.append(acc_test)
		print("\rEvaluating the model: {}/{} ({:.5f}%".format(
			iteration, 
			n_iterations_test, 
			iteration * 100 / n_iterations_test), 
		end = " " * 10)

	loss_test = np.mean(loss_tests)
	acc_test = np.mean(acc_tests)
	print("\nFinal test accuracy: {:.4f}%   Loss: {:.6f}".format(
		acc_test * 100, loss_test))

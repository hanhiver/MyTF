import tensorflow as tf 
import os
import input_data
from matplotlib import pyplot as plt 

mnist = input_data.read_data_sets("../MNIST/MNIST_data/", one_hot = True)

"""
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
"""

# 显示制定当前交互式会话
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sess_conf = tf.ConfigProto()
sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.5
sess_conf.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = sess_conf)

# 权重值初始化函数
# 使用truncated_normal来初始化一个阶段正态分布的权重值，避免零梯度产生。
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

# 偏置值初始化函数
def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

# 卷积函数
def conv2d(x, W):
	return tf.nn.conv2d(
		x, 
		W, 
		strides = [1, 1, 1, 1], 
		padding = 'SAME')

# 池化函数
def max_pool_2x2(x):
	return tf.nn.max_pool(
		x,
		ksize = [1, 2, 2, 1],
		strides = [1, 2, 2, 1],
		padding = 'SAME')

# 设定输入x和输出y_的占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 定义x_image将1x784的输入流转化为28x28的图形
# 第一个参数表示图像数目不确定，最后一个1表示只使用一维的通道
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 将x_image和权值向量进行卷积，然后加上偏置项
# 最后用ReLU激活函数，最后进行max_pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层的卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout层，用来减少过拟合
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
# 使用softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估模型
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float64'))
sess.run(tf.initialize_all_variables())

for i in range(1000):
	batch = mnist.train.next_batch(100)

	if i % 100 == 0:
		train_accurarcy = accuracy.eval(feed_dict = {
			x:batch[0], y_:batch[1], keep_prob:1.0
			})
		print('step {}, training accuracy {}'.format(i, train_accurarcy))

	train_step.run(feed_dict = {
		x:batch[0], y_:batch[1], keep_prob:0.5
		})
"""
print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
"""



	
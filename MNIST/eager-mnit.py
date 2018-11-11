import tensorflow as tf 
import tensorflow.contrib.eager as tfe 
import os
tfe.enable_eager_execution()

import input_data

# mnist = input_data.read_data_sets("./MNIST_data/", one_hot = True)

class MNIST: 
	def __init__(self):
		self.mnist = input_data.read_data_sets("./MNIST_data/", one_hot = True)
		self.train_ds = tf.data.Dataset.from_tensor_slices((self.mnist.train.images, self.mnist.train.labels))\
		.map(lambda x, y : (x, tf.cast(y, tf.float32)))\
		.shuffle(buffer_size=1000).batch(100)

		self.w = tf.get_variable(name = "w", shape = (784, 10))
		self.b = tf.get_variable(name = "b", shape = (10, ))

	def softmax_model(self, image_batch):
		model_output = tf.nn.softmax(tf.matmul(image_batch, self.w) + self.b)
		return model_output

	def cross_entropy(self, model_output, label_batch):
		loss = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(model_output), axis = [1]))
		return loss

	@tfe.implicit_value_and_gradients
	def cal_gradient(self, image_batch, label_batch):
		return self.cross_entropy(self.softmax_model(image_batch), label_batch)

	def train(self):
		optimizer = tf.train.GradientDescentOptimizer(0.5)

		for step, (image_batch, label_batch) in enumerate(tfe.Iterator(self.train_ds)):
			loss, grads_and_vars = self.cal_gradient(image_batch, label_batch)
			optimizer.apply_gradients(grads_and_vars)
			if step % 50 == 0:
				print("Step: {} loss: {}".format(step, loss.numpy()))

	def evaluate(self):
		model_test_output = self.softmax_model(self.mnist.test.images)
		model_test_label = self.mnist.test.labels
		correct_prediction = tf.equal(tf.argmax(model_test_output, 1), tf.argmax(model_test_label, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

		print("Test accuracy = {:.2%}".format(self.accuracy.numpy()))

if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)

	mnist_model = MNIST()
	mnist_model.train()
	mnist_model.evaluate()


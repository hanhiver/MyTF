import tensorflow as tf 
import os
import input_data
from matplotlib import pyplot as plt 


"""
from tensorflow.examples.tutorials.mnist import input_data
"""

mnist = input_data.read_data_sets("./MNIST_data/", one_hot = True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

"""
# Show some example images.
n_samples = 5

plt.figure(figsize = (n_samples * 2, 3))

for index in range(n_samples):
	plt.subplot(1, n_samples, index + 1)
	sample_image = mnist.train.images[index].reshape(28, 28)
	plt.imshow(sample_image, cmap = "binary")
	plt.axis('off')

plt.show()
"""

# Input, a 784 long vector, no number limit. 
x = tf.placeholder(tf.float32, [None, 784])

# Weights, 784 multiple 10 output. 
w = tf.Variable(tf.zeros([784, 10]))

# Bias
b = tf.Variable(tf.zeros([10]))

"""
# Output y, use the softmax model. 
y = tf.nn.softmax(tf.matmul(x, w) + b)

# Caculate the loss. 
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 1))
"""

output = tf.matmul(x, w) + b
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = output))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
#sess = tf.Session(config = config)

sess = tf.InteractiveSession(config = config)

tf.global_variables_initializer().run()

for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
print(sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels}))





import os

# Tensorflow and Keras 
import tensorflow as tf
from tensorflow import keras

# Set the GPU environment. 
# Use 50% gpu memory. 
# import tensorflow.keras.backend.tensorflow_backend as ktf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto(log_device_placement = False)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
keras.backend.set_session(sess)

# Helper libraries
import numpy as np 
import matplotlib.pyplot as plt 
import input_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print(tf.__version__)

# fashion_mnist = keras.datasets.fashion_mnist
fashion_mnist = input_data.read_data_sets("./fashionMNIST/", one_hot = True)

"""
print(fashion_mnist.train.images.shape, fashion_mnist.train.labels.shape)
print(fashion_mnist.test.images.shape, fashion_mnist.test.labels.shape)
print(fashion_mnist.validation.images.shape, fashion_mnist.validation.labels.shape)
"""

train_images = fashion_mnist.train.images.reshape([-1, 28, 28])
train_labels = fashion_mnist.train.labels
train_labels = [ np.argmax(train_labels[i]).item() for i in range(len(train_labels)) ]
test_images = fashion_mnist.test.images.reshape([-1, 28, 28])
test_labels = fashion_mnist.test.labels
test_labels = [ np.argmax(test_labels[i]).item() for i in range(len(test_labels)) ]

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
			'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']


"""
print(test_images.shape)
pr# int(len(test_images))

plt.figure()
plt.imshow(test_images[1])
plt.colorbar()
plt.grid(False)
plt.show()
"""

"""
plt.figure(figsize=(15, 15))
for i in range(25):
	plt.subplot(5, 5, i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i])
	#plt.imshow(train_images[i], cmap = plt.cm.binary)
	plt.colorbar()
	plt.xlabel(class_names[train_labels[i]])
plt.show()
"""

"""
print(train_images.shape)
#print(train_labels)
print(type(train_labels))
print(type(train_labels[0]))
"""

# Setup the layers.
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation = tf.nn.relu),
	keras.layers.Dense(10, activation = tf.nn.softmax)
	])

# Compile the model.
model.compile(
	optimizer = tf.train.AdamOptimizer(),
	loss = 'sparse_categorical_crossentropy',
	metrics = ['accuracy']
	)

# Create a checkpoint callback.
checkpoint_path = 'training1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
	checkpoint_path,
	save_weights_only = True,
	verbose = 1 
	)

# Train the model.
model.fit(train_images, train_labels, epochs = 5,
	validation_data = (test_images, test_labels),
	callbacks = [cp_callback]
	)

# Print the model summary.
model.summary()


# Make prediction.
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f} ({})".format(
		class_names[predicted_label],
		100 * np.max(predictions_array),
		class_names[true_label]
		), color = color)

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color = "#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

"""
i = 12
plt.figure(figsize = (6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()
"""

num_rows = 8
num_cols = 5
num_images = num_rows * num_cols

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
	plt.subplot(num_rows, 2*num_cols, 2 * i + 1)
	plot_image(i, predictions, test_labels, test_images)
	plt.subplot(num_rows, 2*num_cols, 2 * i + 2)
	plot_value_array(i, predictions, test_labels)

plt.show()
import os
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

import numpy as np 

# Set the vocabulary size of the movie review. 
vocab_size = 10000

# Download the IMDB data.
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = vocab_size)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# A dictionary mapping words to an integer index. 
word_index = imdb.get_word_index(path = './imdb_word_index.json')

# The first indices are reserved.
word_index = {k:(v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
	return ' '.join([reverse_word_index.get(i, '?') for i in text])

"""
print(decode_review(train_data[0]))
"""

# Prepare the data.
train_data = keras.preprocessing.sequence.pad_sequences(
	train_data,
	value = word_index["<PAD>"],
	padding = 'post',
	maxlen = 256
	)

test_data = keras.preprocessing.sequence.pad_sequences(
	test_data,
	value = word_index["<PAD>"],
	padding = 'post',
	maxlen = 256
	)

print(len(train_data[0]))
print(len(test_data[1]))

print(train_data[0])

# Build the model.
# Input shape is the vocabulary count used for the movie review (vocab_size words here)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = tf.nn.relu))
model.add(keras.layers.Dense(1, activation = tf.nn.sigmoid))

print(model.summary())

model.compile(
	optimizer = tf.train.AdamOptimizer(), 
	loss = 'binary_crossentropy',
	metrics = ['accuracy']
	)

# Create a validation set. 
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


history = model.fit(
	partial_x_train,
	partial_y_train,
	epochs = 30,
	batch_size = 512,
	validation_data = (x_val, y_val),
	verbose = 1 
	)


# Evaluate the model.
result = model.evaluate(test_data, test_labels)

print(result)

# Create a graph of accuracy and loss over time. 




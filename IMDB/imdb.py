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

print(tf.__version__)


# Download the IMDB data.
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# A dictionary mapping words to an integer index. 
word_index = imdb.get_word_index()

# The first indices are reserved.
word_index = {k:(v + 3) for k, v in word_index()}
word_index["<PAD>"] = 0
word_index["<START"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
	return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])


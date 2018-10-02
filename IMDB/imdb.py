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

"""
# Download the IMDB data.
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 100)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
"""

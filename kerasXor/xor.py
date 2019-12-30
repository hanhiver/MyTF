import os 
import tensorflow as tf  
from tensorflow import keras

from keras.models import load_model
import keras.backend as K
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib


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
from matplotlib import pyplot as plt 

# input data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_labels = np.array([0, 1, 1, 0])

model = keras.Sequential()
model.add(keras.layers.Dense(16, activation = tf.nn.relu, input_dim = 2))
model.add(keras.layers.Dense(16, activation = tf.nn.relu))
model.add(keras.layers.Dense(1, activation = tf.nn.sigmoid))

optimizer = keras.optimizers.SGD(lr = 0.1, decay = 1e-5, momentum = 0.9, nesterov = True)
#optimizer = 'sgd'

loss = 'mse'
#loss = 'mean_absolute_error'

model.compile(
	loss = loss, 
	optimizer = optimizer
	)

"""
history = model.fit(
	x_train, 
	y_labels,
	batch_size = 1, 
	epochs = 1000, 
	shuffle = True,
	verbose = 1, 
	validation_split = 0.0
	)
"""

history = model.fit(
	x_train, 
	y_labels, 
	batch_size = 4,
	epochs = 500, 
	verbose = 2, 
	)

#plt.scatter(range(len(history.history['loss'])), history.history['loss'])
plt.plot(range(len(history.history['loss'])), history.history['loss'])

"""
loss_metrics = model.evaluate(x_train, y_labels, batch_size = 1)
print(loss_metrics)
"""

# Save the model to pb file. 
keras.backend.learning_phase()
saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
checkpoint_path = saver.save(sess, 'saved_ckpt', global_step=0, 
							 latest_filename='checkpoint_state')
graph_io.write_graph(sess.graph, '.', 'tmp.pb')
freeze_graph('./tmp.pb', '', 
			 False, checkpoint_path, "xor_model", 
			 'save/restore_all', 'save/Const:0', 
			 './xor.pb', False, "")
"""
keras.backend.learning_phase()
saver = tf.python.training.saver.Saver(write_version=saver_pb2.SaveDef.v2)
checkpoint_path = saver.save(sess, 'saved_ckpt', global_step=0, 
							 latest_filename='checkpoint_state')
tf.python.framework.graph_io.write_graph(sess.graph, '.', 'tmp.pb')
tf.python.tools.freeze_graph('./tmp.pb', '', 
							 False, checkpoint_path, out_names, 
							 'save/restore_all', 'save/Const:0', 
							 models_dir+model_filename, False, "")
"""

print(model.predict(x_train))
plt.show()

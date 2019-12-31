from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras 

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

import numpy as np
import matplotlib.pyplot as plt 

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) \
    = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_labels))

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), 
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3)

model.summary()

print("Save model. ")

#model.save_weights('./checkpoints/my_checkpoint')
model.save_weights('saved_model/my_weight')
model.save('saved_model/my_model.pb')

"""
K.set_learning_phase(0)
sess = K.get_session()
saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
checkpoint_path = saver.save(sess, 
                            'saved_ckpt', 
                            global_step=0, 
                            latest_filename='checkpoint_state')

graph_io.write_graph(sess.graph, '.', 'tmp.pb')
freeze_graph.freeze_graph('./tmp.pb', '',
                      	False, checkpoint_path, out_names,
                      	"save/restore_all", "save/Const:0",
                      	models_dir+model_filename, False, "")
"""

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy: ', test_acc)

predictions = model.predict(test_images)

def saveModel(self, export_version, export_path = 'prod_model'):
    print('==S== Save Model: version {} to {} ==S=='.format(export_version, export_path))
    print(type(self.yolo_model))
    print("==S== Model Input: {}".format(self.yolo_model.input))
    print("==S== Model Output: {}".format(self.yolo_model.output))
    #print("==S== Model Summary: ")
    self.yolo_model.summary()

    """
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'image': self.yolo_model.input}, 
        outputs={'boxes': self.yolo_model.output[0], 
                    'scores': self.yolo_model.output[1], 
                    'classes': self.yolo_model.output[2]})
    
    """
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'image': self.yolo_model.input, 
                'image_shape': self.input_image_shape, 
                'learning_phase': tf.constant(0, dtype = tf.float32)}, 
        outputs={'boxes': self.boxes, 
                    'scores': self.scores, 
                    'classes': self.classes})
    

    export_path = os.path.join(tf.compat.as_bytes(export_path), 
                                tf.compat.as_bytes(str(export_version)))

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    
    builder.add_meta_graph_and_variables(
                sess = self.sess,
                tags = [tf.saved_model.tag_constants.SERVING],
                signature_def_map = {'high_way_detector': signature, },
                legacy_init_op = legacy_init_op)

    #builder.save()
    print('==S== Model saved SUCCESSFULLY!!! ')

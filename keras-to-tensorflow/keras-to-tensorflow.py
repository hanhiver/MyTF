import sys 
import logging

import tensorflow as tf 
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from pathlib import Path 

import keras 
from keras import backend as K 
from keras.models import model_from_json, model_from_yaml

K.set_learning_phase(0)

def load_model(input_model_path):
    if not Path(input_model_path).exists():
        raise FileNotFoundError(
                'Model file {} does not exist.'.format(input_model_path))
    
    try:
        model = keras.models.load_model(input_model_path)
        return model
    except FileNotFoundError as err:
        logging.error('Input mode file ({}) does not exist.'.format(input_model_path))
        raise err
    except ValueError as wrong_file_err:
        logging.error('Failed to load the model file. ')
        raise wrong_file_err

def main(quantize=False):
    model = load_model(sys.argv[1])
    model.summary()

    orig_output_node_names = [node.op.name for node in model.outputs]
    logging.info('Converted output node names are: {}'.format(str(orig_output_node_names)))

    sess = K.get_session()
    saver = tf.train.Saver()
    saver.save(sess, "./saved_ckpt/")

    if quantize:
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ['quantize_weights', 'quantize_nodes']
        transform_graph_def = TransformGraph(sess.graph.as_graph_def(), [], 
                                            orig_output_node_names, 
                                            transforms)
    else:
        transform_graph_def = sess.graph.as_graph_def()

    constant_graph = graph_util.convert_variables_to_constants(
                        sess,
                        transform_graph_def, 
                        orig_output_node_names)

    graph_io.write_graph(constant_graph, "./saved_ckpt/", "xor.pb", as_text=False) 

if __name__ == "__main__":
    main()
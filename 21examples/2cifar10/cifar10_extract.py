# coding: utf-8
import cifar10_input

import tensorflow as tf 
import os 
import scipy.misc

def input_origin(data_dir):
	filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
				for i in range(1, 6)]

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	filename_queue = tf.train.string_input_producer(filenames)
	read_input = cifar10_input.read_cifar10(filename_queue)

	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	return reshaped_image

if __name__ == '__main__':

	with tf.Session() as sess:
		reshaped_image = input_origin('cifar10_data/cifar-10-batches-bin')

		threads = tf.train.start_queue_runners(sess = sess)

		sess.run(tf.global_variables_initializer())

		if not os.path.exists('cifar10_data/raw/'):
			os.makedirs('cifar10_data/raw/')

		for i in range(30): 
			image_array = sess.run(reshaped_image)
			scipy.misc.toimage(image_array).save('cifar10_data/raw/%d.jpg' % i)


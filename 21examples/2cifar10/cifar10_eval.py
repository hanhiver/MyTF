from datetime import datetime
import math
import time 

import numpy as np 
import tensorflow as tf 

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval', 
	""" Directory of where to write the event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test', 
	"""Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
	"""Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
	"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000, 
	"""Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
	"""Whether to run eval only once""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
	""" 一次性评价

	参数：
		saver: Saver. 
		summary_writer: Summary writer. 
		top_k_op: Top K op. 
		summary_op: Summary op. 
	"""

	with tf.Session() as sess: 
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path: 
			# 从checkpoint中读取训练好的模型
			saver.restore(sess, ckpt.model_checkpoint_path)

			# 假设model_checkpoint_path大概是如下样子：
			# /my-favorite-path/cifar10_train/model.ckpt-0
			# 从模型的checkpoint中解压出global_step。
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('没有找到训练的checkpoint数据')
			return 

		# 启动queue runners. 
		coord = tf.train.Coordinator()

		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord = coord, daemon = True, start = True))

			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			step = 0
			while step < num_iter and not coord.should_stop():
				predictions = sess.run([top_k_op])
				true_count += np.sum(predictions)
				step += 1

			# 计算预测准确率
			precision = true_count / total_sample_count
			print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag = 'Precision @ 1', simple_value = precision)
			summary_writer.add_summary(summary, global_step)
		except Exception as e: # pylint: disable=broad-except
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs = 10)

def evaluate():
	""" 多次评价CIFAR-10 """
	with tf.Graph().as_default() as g: 
		# 从CIFAR-10中读取图像和标签
		eval_data = FLAGS.eval_data == 'test'
		images, labels = cifar10.inputs(eval_data = eval_data)

		# 构建一个图计算推理模型的logits
		logits = cifar10.inference(images)

		# 计算准确度
		top_k_op = tf.nn.in_top_k(logits, labels, 1)

		# 读取已经训练好的模型参数
		variable_averages = tf.train.ExponentialMovingAverage(
			cifar10.MOVING_AVERAGE_DECAY)
		variable_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		# 用TF collection of Summaries构建summary operation
		summary_op = tf.summary.merge_all()

		summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

		while True: 
			eval_once(saver, summary_writer, top_k_op, summary_op)
			if FLAGS.run_once:
				break
			time.sleep(FLAGS.eval_interval_secs)

def main(argv = None): # pylint: disable=unused-argument
	cifar10.maybe_download_and_extract()
	if tf.gfile.Exists(FLAGS.eval_dir):
		tf.gfile.DeleteRecursively(FLAGS.eval_dir)
	tf.gfile.MakeDirs(FLAGS.eval_dir)
	evaluate()

if __name__ == '__main__':
	tf.app.run()











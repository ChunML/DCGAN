from model import DCGAN
import tensorflow as tf
import os
import scipy.misc
import numpy as np

flags = tf.app.flags
flags.DEFINE_integer('epoch', 25, 'Training epochs')
flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate for Adam Optimizer')
flags.DEFINE_float('beta1', 0.5, 'Beta1 for Adam Optimizer')
flags.DEFINE_integer('train_size', np.inf, 'Size of training images')
flags.DEFINE_integer('batch_size', 64, 'Training batch size')
flags.DEFINE_integer('input_height', 28, 'Input image height')
flags.DEFINE_integer('input_width', 28, 'Input image width')
flags.DEFINE_integer('output_height', 28, 'Cropped image height')
flags.DEFINE_integer('output_width', 28, 'Cropped image width')
flags.DEFINE_integer('c_dim', 3, 'Dimension of (colored) image')
flags.DEFINE_string('dataset', 'mnist', 'Name of dataset')
flags.DEFINE_string('input_fname_pattern', '*.jpg', 'Extension of input images')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'Folder to save checkpoints')
flags.DEFINE_string('sample_dir', 'samples', 'Folder to save sampled images')
flags.DEFINE_boolean('is_train', True, 'True for training, False for testing')
flags.DEFINE_boolean('is_crop', False, 'True for training, False for testing')
flags.DEFINE_boolean('visualizer', False, 'True for visualizing, False for nothing')
FLAGS = flags.FLAGS

def main(_):
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)

	with tf.Session() as sess:
		dcgan = DCGAN(
			sess,
			input_height=FLAGS.input_height,
			input_width=FLAGS.input_width,
			output_height=FLAGS.output_height,
			output_width=FLAGS.output_width,
			batch_size=FLAGS.batch_size,
			sample_num=FLAGS.batch_size,
			c_dim=1,
			dataset_name=FLAGS.dataset,
			input_fname_pattern=FLAGS.input_fname_pattern,
			is_crop=FLAGS.is_crop,
			checkpoint_dir=FLAGS.checkpoint_dir,
			sample_dir=FLAGS.sample_dir)

		if FLAGS.is_train:
			dcgan.train(FLAGS)
		else:
			print('Loading checkpoint hasn\'t been implemented yet.')

if __name__ == '__main__':
	tf.app.run()

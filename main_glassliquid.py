""" Deep Learning for glass-vs-liquid data """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, argparse
import math
import time
import os
import tensorflow as tf
import numpy as np
import json
from generator_glassliquid import Generator
from collections import Counter
np.random.seed(0)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Function to parse arguments
def create_parser():

	# Create parser and add arguments
	parser = argparse.ArgumentParser(description='Read hyperparameter information')
	parser.add_argument('-eta', dest='eta', default=1e-3, help='Learning rate')
	parser.add_argument('-batch_size', dest='batch_size', default=10, help='Batch size')
	parser.add_argument('-epochs', dest='epochs', default=100, help='Number of epochs to run')
	parser.add_argument('-beta', dest='beta', default=0.01, help='L2 regularization parameter')
	parser.add_argument('-keep_probability', dest='keep_probability', default=0.5, help='Dropout keep probability')

	return parser

# Function to convert arguments into a dictionary
def convert_args(args):

	# Options dictionary
	options = {}
	options['eta'] = args.eta
	options['batch_size'] = args.batch_size
	options['epochs'] = args.epochs
	options['beta'] = args.beta
	options['keep_probability'] = args.keep_probability

	return options

""" Functions to convert the labels to one hot """
def get_unique_labels(metadata):
	return list(set([row['label'] for row in metadata]))

def create_one_hot_mapping(unique_labels):
	one_hot_mapping = dict()

	for i, label in enumerate(unique_labels):
		one_hot = np.zeros(len(unique_labels))
		one_hot[i] = 1
		one_hot_mapping[label] = one_hot

	return one_hot_mapping

def convert_to_one_hot(metadata, one_hot_mapping):
	for row in metadata:
		row['original_label'] = row['label']
		row['label'] = one_hot_mapping[row['label']]

	return metadata

def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, name):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=0.1, seed=0)
	return tf.Variable(initial, name=name)


def bias_variable(shape, name):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

""" Function that builds the graph for the neural network """
def deepnn(x, n_outputs):
	# First convolutional layer
	x_image = tf.reshape(x, [-1, 250, 250, 3])
	W_conv1 = weight_variable([10, 10, 3, 6], name="W_conv1")
	b_conv1 = bias_variable([6], name="b_conv1")
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

	# Second convolutional layer
	W_conv2 = weight_variable([5, 5, 6, 16], name="W_conv2")
	b_conv2 = bias_variable([16], name="b_conv2")
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

	# Fully connected layer
	W_fc1 = weight_variable([237 * 237 * 16, 80], name="W_fc1")
	b_fc1 = bias_variable([80], name="b_fc1")
	h_conv2_flat = tf.reshape(h_conv2, [-1, 237*237*16])
	h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

	# Dropout on the fully connected layer
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, seed=0)

	# Output layer
	W_fc2 = weight_variable([80, n_outputs], name="W_fc2")
	b_fc2 = bias_variable([n_outputs], name="b_fc2")
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	# Returns the prediction and the dropout probability placeholder
	return y_conv, keep_prob, W_conv1, W_conv2, W_fc1, W_fc2


def plot(train_accuracies, train_losses, validation_accuracies, validation_losses, eta_original, batch_size, beta, keep_probability):
	plt.subplot(221)
	plt.plot(range(len(train_losses)), train_losses)
	plt.title("Training")
	plt.ylabel('Loss')

	plt.subplot(222)
	plt.plot(range(len(validation_losses)), validation_losses)
	plt.title("validation")

	plt.subplot(223)
	plt.plot(range(len(train_accuracies)), train_accuracies)
	plt.ylabel('Accuracy')
	plt.xlabel('Number of epochs')

	plt.subplot(224)
	plt.plot(range(len(validation_accuracies)), validation_accuracies)
	plt.xlabel('Number of epochs')

	plt.savefig('learning_rate_test'+'-'+str(eta_original)+'-'+str(batch_size)+'-'+str(beta)+'-'+str(keep_probability)+'.png')

def main(argv):

	parser = create_parser()
	args = parser.parse_args()
	options = convert_args(args)

	""" Load the metadata """
	with open('metadata/metadata.json', 'r') as f:
		metadata = json.load(f)

	""" Shuffle the data """
	np.random.shuffle(metadata)

	""" Convert the data to one hot """
	unique_labels = get_unique_labels(metadata)
	one_hot_mapping = create_one_hot_mapping(unique_labels)
	metadata = convert_to_one_hot(metadata, one_hot_mapping)

	""" Define input and output sizes """
	im_size = 250
	n_outputs = len(unique_labels)

	""" Create batch generators for train and test """
	train_metadata, remaining_metadata = train_test_split(metadata, test_size=0.3, random_state=0)
	validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size = 0.5, random_state = 0)

	train_generator = Generator(train_metadata, im_size=im_size, num_channel=4, data_aug=False)
	validation_generator = Generator(validation_metadata, im_size=im_size, num_channel=4, data_aug=False)
	test_generator = Generator(test_metadata, im_size=im_size, num_channel=4, data_aug=False)

	""" Hyperparameters """
	batch_size = int(options['batch_size'])
	epochs = int(options['epochs'])
	batches_per_epoch = 10
	examples_per_eval = 1000
	eta_original = float(options['eta'])
	beta = float(options['beta'])
	keep_probability = float(options['keep_probability'])

	# Input data
	x = tf.placeholder(tf.float32, [None, im_size, im_size, 3])

	# Output
	y_ = tf.placeholder(tf.float32, [None, n_outputs])

	# Build the graph for the deep net
	y_conv, keep_prob, W_conv1, W_conv2, W_fc1, W_fc2 = deepnn(x, n_outputs)

	# Learning rate
	learning_rate = tf.placeholder(tf.float32, shape=[])

	# Define the los and the optimizer
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
	loss = tf.reduce_mean(loss + beta*regularizers)
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Save GPU memory preferences
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	""" Lists for plotting """
	train_losses = []
	train_accuracies = []
	validation_losses = []
	validation_accuracies = []

	# Saver
	saver = tf.train.Saver()

	# Accuracy tracker
	accuracy_tracker = 0.0

	# Update number tracker
	update_tracker = 0

	# Run the network
	with tf.Session(config=config) as sess:

		# Initialize variables
		sess.run(tf.global_variables_initializer())

		# Print class balance
		train_counts = Counter(row['original_label'] for row in train_generator.metadata)
		validation_counts = Counter(row['original_label'] for row in validation_generator.metadata)
		test_counts = Counter(row['original_label'] for row in test_generator.metadata)

		print('')
		print('class balance')
		print('train counts')
		print(train_counts)
		print('validation counts')
		print(validation_counts)
		print('test counts')
		print(test_counts)
		print('')

		# Print hyperparameters
		print('epochs = %d, eta = %g, batch_size = %g, beta = %g, keep_probability = %g' % (epochs, eta_original, batch_size, beta, keep_probability))
		print('')

		# Training
		print('Training')
		for epoch in range(epochs):
			print('epoch {}'.format(epoch))
			print('Evaluating')

			# Evaluate on train set
			train_batch_accuracies = []
			train_batch_losses = []
			for train_X, train_Y in train_generator.data_in_batches(examples_per_eval, batch_size):
				train_batch_accuracies.append(accuracy.eval(feed_dict={
						x: train_X, y_: train_Y, keep_prob: 1.0}))

				train_batch_losses.append(loss.eval(feed_dict={
						x: train_X, y_: train_Y, keep_prob: 1.0}))

			train_accuracy = np.mean(train_batch_accuracies)
			train_loss = np.mean(train_batch_losses)

			train_accuracies.append(train_accuracy)
			train_losses.append(train_loss)

			# Evaluate on validation set
			validation_batch_accuracies = []
			validation_batch_losses = []
			for validation_X, validation_Y in validation_generator.data_in_batches(examples_per_eval, batch_size):
				validation_batch_accuracies.append(accuracy.eval(feed_dict={
						x: validation_X, y_: validation_Y, keep_prob: 1.0}))

				validation_batch_losses.append(loss.eval(feed_dict={
						x: validation_X, y_: validation_Y, keep_prob: 1.0}))

			validation_accuracy = np.mean(validation_batch_accuracies)
			validation_loss = np.mean(validation_batch_losses)

			validation_accuracies.append(validation_accuracy)
			validation_losses.append(validation_loss)

			print('training accuracy %g, train loss %g, ' \
				'validation accuracy %g, validation loss %g' %
				(train_accuracy, train_loss, validation_accuracy, validation_loss))
			print('')

			plot(train_accuracies, train_losses, validation_accuracies, validation_losses, eta_original, batch_size, beta, keep_probability)


			# Save the best model
			if validation_accuracy > accuracy_tracker:

				saver.save(sess, 'learning_rate_test'+'-'+str(eta_original)+'-'+str(batch_size)+'-'+str(beta)+'-'+str(keep_probability)+'.ckpt')

				accuracy_tracker = validation_accuracy

			# Train
			# for i in tqdm(range(batches_per_epoch)):
			for i in range(batches_per_epoch):

				# Use a learning rate schedule for the first 100 updates
				if update_tracker < 100:
					eta = (1.0 - float(update_tracker)/100.0)*0.001 + (float(update_tracker)/100.0)*eta_original
					train_X, train_Y = train_generator.next(batch_size)
					train_step.run(feed_dict={x: train_X, y_: train_Y, keep_prob: keep_probability, learning_rate: eta})
				else:
					eta = eta_original
					train_X, train_Y = train_generator.next(batch_size)
					train_step.run(feed_dict={x: train_X, y_: train_Y, keep_prob: keep_probability, learning_rate: eta})
				print(eta)
				update_tracker += 1

		# Compute total validation set error
		validation_total_accuracies = []
		for validation_X, validation_Y in validation_generator.data_in_batches(len(validation_generator.metadata), batch_size):
			validation_total_accuracies.append(accuracy.eval(feed_dict={
					x: validation_X, y_: validation_Y, keep_prob: 1.0}))

		validation_accuracy = np.mean(validation_total_accuracies)

		print('final validation accuracy %g' % (validation_accuracy))

		# Compute total test set error
		test_total_accuracies = []
		for test_X, test_Y in test_generator.data_in_batches(len(test_generator.metadata), batch_size):
			test_total_accuracies.append(accuracy.eval(feed_dict={
					x: test_X, y_: test_Y, keep_prob: 1.0}))

		test_accuracy = np.mean(test_total_accuracies)

		print('final test accuracy %g' % (test_accuracy))

		



	plot(train_accuracies, train_losses, validation_accuracies, validation_losses, eta_original, batch_size)

	# train_counts = Counter(row['original_label'] for row in train_generator.metadata)
	# validation_counts = Counter(row['original_label'] for row in validation_generator.metadata)
	# test_counts = Counter(row['original_label'] for row in test_generator.metadata)

	# f = open('batch'+'-'+str(eta_original)+'-'+str(batch_size)+'.txt', 'w')
	# f.write('final validation accuracy %g \n' % (validation_accuracy))
	# f.write('test accuracy %g \n' % (test_accuracy))
	# f.write('train counts glass %g, liquid %g \n' % (train_counts['glass'], train_counts['liquid']))
	# f.write('validation counts glass %g, liquid %g \n' % (validation_counts['glass'], validation_counts['liquid']))
	# f.write('test counts glass %g, liquid %g \n' % (test_counts['glass'], test_counts['liquid']))
	# f.write('epochs = %d, eta = %g, batch_size = %g' % (epochs, eta_original, batch_size))
	# f.close()

# Run the program
if __name__ == '__main__':
	tf.app.run(main=main(sys.argv[1:]))

""" Deep Learning for Molecular Modeling Project """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np 
import json
from generator import Generator
from collections import Counter
np.random.seed(0)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from tqdm import tqdm

""" Load the metadata """
with open('metadata/metadata.json', 'r') as f:
	metadata = json.load(f)

""" Restrict data """
train_points = [[0.4, 0.05], [0.4, 0.10], [0.4, 0.60], [0.4, 0.70], [0.5, 0.05], [0.5, 0.10], [0.5, 0.60], [0.5, 0.70],
[0.7, 0.05], [0.7, 0.10], [0.7, 0.60], [0.7, 0.70], [0.8, 0.05], [0.8, 0.10], [0.8, 0.70]]
test_points = [[0.8, 0.60]]

""" Shuffle the data """
np.random.shuffle(metadata)

train_metadata = [row for row in metadata if row['label'] in train_points]
final_test_metadata = [row for row in metadata if row['label'] in test_points]

for row in train_metadata:
	row['label'] = tuple(row['label'])

for row in final_test_metadata:
	row['label'] = tuple(row['label'])

""" Define input and output sizes """
im_size = 250

""" Create batch generators for train and test """
train_metadata, test_metadata = train_test_split(train_metadata, test_size=0.2, random_state=0)

train_generator = Generator(train_metadata, im_size=im_size)
test_generator = Generator(test_metadata, im_size=im_size)
final_test_generator = Generator(final_test_metadata, im_size=im_size)

""" Hyperparameters """
batch_size = 200
epochs = 10
batches_per_epoch = 10
examples_per_eval = 1000
eta = 1e-4
beta = 0.01

""" Function that builds the graph for the neural network """
def deepnn(x):
	# First convolutional layer
	x_image = tf.reshape(x, [-1, 250, 250, 1])
	W_conv1 = weight_variable([10, 10, 1, 12])
	b_conv1 = bias_variable([12])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
 
	# Second convolutional layer
	W_conv2 = weight_variable([5, 5, 12, 32])
	b_conv2 = bias_variable([32])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# Fully connected layer for T
	W_fc1T = weight_variable([59 * 59 * 32, 80])
	b_fc1T = bias_variable([80])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 59*59*32])
	h_fc1T = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1T) + b_fc1T)

	# Fully connected layer for rho
	W_fc1rho = weight_variable([59 * 59 * 32, 80])
	b_fc1rho = bias_variable([80])
	h_fc1rho = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1rho) + b_fc1rho)

	# Dropout on the fully connected T layer
	keep_prob = tf.placeholder(tf.float32)
	h_fc1T_drop = tf.nn.dropout(h_fc1T, keep_prob, seed=0)

	# Dropout on the fully connected rho layer
	h_fc1rho_drop = tf.nn.dropout(h_fc1rho, keep_prob, seed=0)

	# Output for T
	W_fc2T = weight_variable([80, 1])
	b_fc2T = bias_variable([1])
	T = tf.matmul(h_fc1T_drop, W_fc2T) + b_fc2T

	# Output for rho
	W_fc2rho = weight_variable([80, 1])
	b_fc2rho = bias_variable([1])
	rho = tf.matmul(h_fc1rho_drop, W_fc2rho) + b_fc2rho

	# Returns the prediction and the dropout probability placeholder
	return T, rho, W_conv1, W_conv2, W_fc1T, W_fc1rho, W_fc2T, W_fc2rho, keep_prob


def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=0.1, seed=0)
	return tf.Variable(initial)


def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def plot(train_accuracies, train_losses, test_accuracies, test_losses):
	plt.subplot(221)
	plt.plot(range(len(train_losses)), train_losses)
	plt.title("Training")
	plt.ylabel('Loss')
	
	plt.subplot(222)
	plt.plot(range(len(test_losses)), test_losses)
	plt.title("Test")
			
	plt.subplot(223)
	plt.plot(range(len(train_accuracies)), train_accuracies)
	plt.ylabel('Accuracy')
	plt.xlabel('Number of epochs')

	plt.subplot(224)
	plt.plot(range(len(test_accuracies)), test_accuracies)
	plt.xlabel('Number of epochs')

	plt.savefig('regression.png')

def main(_):
	# Input data
	x = tf.placeholder(tf.float32, [None, im_size*im_size])

	# Output
	y_T = tf.placeholder(tf.float32, [None, 1])
	y_rho = tf.placeholder(tf.float32, [None, 1])

	# Build the graph for the deep net
	T, rho, W_conv1, W_conv2, W_fc1T, W_fc1rho, W_fc2T, W_fc2rho, keep_prob = deepnn(x)

	# Define the los and the optimizer
	T_loss = tf.nn.l2_loss(y_T - T)
	rho_loss = tf.nn.l2_loss(y_rho - rho)
	loss = T_loss + rho_loss
	regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1T) + tf.nn.l2_loss(W_fc1rho) + tf.nn.l2_loss(W_fc2T) + tf.nn.l2_loss(W_fc2rho)
	loss = tf.reduce_mean(loss + beta*regularizers)
	train_step = tf.train.AdamOptimizer(eta).minimize(loss)
	
	# Save GPU memory preferences
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	""" Lists for plotting """
	train_accuracies = []
	train_losses = []
	test_accuracies = []
	test_losses = []
	
	# Run the network
	with tf.Session(config=config) as sess:

		# Initialize variables
		sess.run(tf.global_variables_initializer())

		# Print class balance
		train_counts = Counter(row['label'] for row in train_generator.metadata)
		test_counts = Counter(row['label'] for row in test_generator.metadata)

		print('')
		print('class balance')
		print('train counts')
		print(train_counts)
		print('test counts')
		print(test_counts)
		print('')

		# Print hyperparameters
		print('epochs = %d, eta = %g, batch_size = %g' % (epochs, eta, batch_size))
		print('train_points')
		print(train_points)
		print('test_points')
		print(test_points)
		print('')

		# Training
		print('Training')
		for epoch in range(epochs):
			print('epoch {}'.format(epoch))
			print('Evaluating')

			if(epoch > 5):
			
				# Evaluate on train set
				train_batch_losses = []
				for train_X, train_Y in train_generator.data_in_batches(examples_per_eval, batch_size):
					train_T = np.zeros((len(train_Y), 1))
					train_rho = np.zeros((len(train_Y), 1))
					for i in range(len(train_Y)):
						train_T[i][0] = train_Y[i][0]
						train_rho[i][0] = train_Y[i][1]
					train_batch_losses.append(loss.eval(feed_dict={x: train_X, y_T: train_T, y_rho: train_rho, keep_prob: 1.0}))
					#print(T.eval(feed_dict={x: train_X, y_T: train_T, y_rho: train_rho, keep_prob: 1.0}))

				train_loss = np.mean(train_batch_losses)

				train_losses.append(train_loss)

				# Evaluate on test set
				test_batch_losses = []
				for test_X, test_Y in test_generator.data_in_batches(examples_per_eval, batch_size):
					test_T = np.zeros((len(test_Y), 1))
					test_rho = np.zeros((len(test_Y), 1))
					for i in range(len(test_Y)):
						test_T[i][0] = test_Y[i][0]
						test_rho[i][0] = test_Y[i][1]
					test_batch_losses.append(loss.eval(feed_dict={x: test_X, y_T: test_T, y_rho: test_rho, keep_prob: 1.0}))

				test_loss = np.mean(test_batch_losses)

				test_losses.append(test_loss)

				print('train loss %g, ' \
					'validation loss %g' %
					(train_loss, test_loss))
				print('')

				plot(train_accuracies, train_losses, test_accuracies, test_losses)
			
			# Train
			# for i in tqdm(range(batches_per_epoch)):
			for i in range(batches_per_epoch):
				train_X, train_Y = train_generator.next(batch_size)
				train_T = np.zeros((len(train_Y), 1))
				train_rho = np.zeros((len(train_Y), 1))
				for i in range(len(train_Y)):
					train_T[i][0] = train_Y[i][0]
					train_rho[i][0] = train_Y[i][1]
				train_step.run(feed_dict={x: train_X, y_T: train_T, y_rho: train_rho, keep_prob: 0.5})

		final_test_batch_losses = []
		final_test_batch_T = []
		final_test_batch_rho = []
		for i in range(5):
			test_X, test_Y = final_test_generator.next(batch_size)
			test_T = np.zeros((len(test_Y), 1))
			test_rho = np.zeros((len(test_Y), 1))
			for i in range(len(test_Y)):
				test_T[i][0] = test_Y[i][0]
				test_rho[i][0] = test_Y[i][1]
			final_test_batch_losses.append(loss.eval(feed_dict={x: test_X, y_T: test_T, y_rho: test_rho, keep_prob: 1.0}))
			final_test_batch_T.append(T.eval(feed_dict={x: test_X, y_T: test_T, y_rho: test_rho, keep_prob: 1.0}))
			final_test_batch_rho.append(rho.eval(feed_dict={x: test_X, y_T: test_T, y_rho: test_rho, keep_prob: 1.0}))
		print(np.mean(final_test_batch_losses))
		print(np.mean(final_test_batch_T))
		print(np.mean(final_test_batch_rho))
		file = open("prediction.txt", "w")
		file.write("Loss: ")
		file.write(str(np.mean(final_test_batch_losses)))
		file.write('\n')
		file.write("T: ")
		file.write(str(np.mean(final_test_batch_T)))
		file.write('\n')
		file.write("rho: ")
		file.write(str(np.mean(final_test_batch_rho)))

	plot(train_accuracies, train_losses, test_accuracies, test_losses)

# Run the program 
if __name__ == '__main__':
	tf.app.run(main=main)

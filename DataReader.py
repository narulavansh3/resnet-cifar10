import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
	"""Load the CIFAR-10 dataset.

	Args:
		data_dir: A string. The directory where data batches
			are stored.

	Returns:
		x_train: An numpy array of shape [50000, 3072].
			(dtype=np.float32)
		y_train: An numpy array of shape [50000,].
			(dtype=np.int32)
		x_test: An numpy array of shape [10000, 3072].
			(dtype=np.float32)
		y_test: An numpy array of shape [10000,].
			(dtype=np.int32)
	"""

	### YOUR CODE HERE
	


## Number of channels in each image, 3 channels: Red, Green, Blue.
#	num_channels = 3
	
#	#_num_files_train = 5
	
	
	raw_images = []
	raw_class = []
	raw_test_images = []
	raw_test_classes = []
	
	
	for i in range(1,6):
		# Load the images and class-numbers from the data-file.
		#filename="data_batch_" + str(i)
	
		with open(data_dir+"data_batch_" + str(i),"rb") as f:
			data = pickle.load(f,encoding='bytes')
		raw_images.extend(data[b'data'])
		raw_class.extend(data[b'labels'])

	x_train = np.array(raw_images, dtype=float) / 255.0
	y_train = np.array(raw_class)	
	with open(data_dir + "/test_batch", 'rb') as fo:
		test_data_dict = pickle.load(fo, encoding='bytes')
	raw_test_images.extend(data[b'data'])
	raw_test_classes.extend(data[b'labels'])
	x_test = np.array(raw_test_images, dtype=float) / 255.0
	y_test = np.array(raw_test_classes)

	
	### END CODE HERE

	return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, split_index=45000):
	"""Split the original training data into a new training dataset
	and a validation dataset.

	Args:
		x_train: An array of shape [50000, 3072].
		y_train: An array of shape [50000,].
		split_index: An integer.

	Returns:
		x_train_new: An array of shape [split_index, 3072].
		y_train_new: An array of shape [split_index,].
		x_valid: An array of shape [50000-split_index, 3072].
		y_valid: An array of shape [50000-split_index,].
	"""
	x_train_new = x_train[:split_index]
	y_train_new = y_train[:split_index]
	x_valid = x_train[split_index:]
	y_valid = y_train[split_index:]

	return x_train_new, y_train_new, x_valid, y_valid


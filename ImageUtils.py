import numpy as np

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
	"""Parse a record to an image and perform data preprocessing.

	Args:
		record: An array of shape [3072,]. One row of the x_* matrix.
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [32, 32, 3].
	"""
	# Reshape from [depth * height * width] to [depth, height, width].
	# depth_major = tf.reshape(record, [3, 32, 32])
	depth_major = record.reshape((3, 32, 32))

	# Convert from [depth, height, width] to [height, width, depth]
	# image = tf.transpose(depth_major, [1, 2, 0])
	image = np.transpose(depth_major, [1, 2, 0])

	image = preprocess_image(image, training)
	
	#print('shape of image in data augmentation', image.shape)
	return image


def preprocess_image(image, training):
	"""Preprocess a single image of shape [height, width, depth].

	Args:
		image: An array of shape [32, 32, 3].
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [32, 32, 3].
	"""
	if training:
		### YOUR CODE HERE
		#import tensorflow as tf
		# Resize the image to add four extra pixels on each side.
		image = np.stack([np.pad(image[:,:,c],((4,4),(4,4)),mode='constant',constant_values=0) for c in range(3)], axis =2)

		### END CODE HERE

		### YOUR CODE HERE
		# Randomly crop a [32, 32] section of the image.
		#image = tf.random_crop(image, [32, 32, 3])
		# HINT: randomly generate the upper left point of the image
		import random
		x = random.randint(0,8)
		y = random.randint(0,8)
		image = np.stack([image[x:x+32,y:y+32,c] for c in range(3)], axis =2)

		### END CODE HERE

		### YOUR CODE HERE
		# Randomly flip the image horizontally.
		#image = tf.image.random_flip_left_right(image)
		flip = random.randint(0,1)
		if flip:
			image = np.stack([np.fliplr(image[:,:,c]) for c in range(3)], axis =2)
		#print('after', image)
		
		### END CODE HERE

	### YOUR CODE HERE
	# Subtract off the mean and divide by the variance of the pixels.
	#image = tf.image.per_image_standardization(image)
	#image = image/256
	image = (image-image.mean())/image.std()
	
	### END CODE HERE

	return image


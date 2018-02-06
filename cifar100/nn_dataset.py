# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# I modified the code. I added the option for 'num_epochs=None' to repeat indefinitely during training or validation.
# Added the _LABEL variable to define the number of bytes in the record to extend to cifar100.
# Added more data augmentation such as: rot90, random order for brightness and contrast.

import tensorflow as tf
import os
from tarfile import open as tar_open
from urllib import request as urllib_req
from sys import stdout

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_LABEL = 2
NUM_CLASSES = 100

NUM_IMAGES = {
	'train': 50000,
	'validation': 10000,
}


def maybe_download_and_extract(base_dir):
	"""Download and extract the tarball from Alex's website."""
	DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(base_dir, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count*block_size) / float(total_size)*100.0))
			stdout.flush()

		filepath, _ = urllib_req.urlretrieve(DATA_URL, filepath, _progress)
		print()
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	extracted_dir_path = os.path.join(base_dir, 'cifar-100-binary')
	if not os.path.exists(extracted_dir_path):
		tar_open(filepath, 'r:gz').extractall(base_dir)


def get_filenames(is_training, data_dir):
	"""Returns a list of filenames."""
	data_dir = os.path.join(data_dir, 'cifar-100-binary')

	assert os.path.exists(data_dir), (
		'Run cifar100_download_and_extract.py first to download and extract the '
		'CIFAR-100 data.')

	if is_training:
		return [os.path.join(data_dir, 'train.bin')]
	else:
		return [os.path.join(data_dir, 'test.bin')]


def record_dataset(filenames):
	"""Returns an input pipeline Dataset from `filenames`."""
	record_bytes = _HEIGHT * _WIDTH * _DEPTH + _LABEL
	return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def preprocess_image(image, is_training):
	"""Preprocess a single image of layout [height, width, depth]."""
	if is_training:
		def rand_brightness_contrast(order, im):
			if order == 0:
				im = tf.image.random_brightness(im, max_delta=63)
				return tf.image.random_contrast(im, lower=0.2, upper=1.8)
			else:
				im = tf.image.random_contrast(im, lower=0.2, upper=1.8)
				return tf.image.random_brightness(im, max_delta=63)

		# Resize the image to add four extra pixels on each side.
		image = tf.image.resize_image_with_crop_or_pad(image, _HEIGHT+8, _WIDTH+8)

		# Randomly crop a [_HEIGHT, _WIDTH] section of the image.
		image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

		# Randomly flip the image horizontally.
		image = tf.image.random_flip_left_right(image)
		# Randomly rotate the image counter clockwise by 90 degrees. k=0 no rot up-to k=3.
		# image = tf.image.rot90(image, k=tf.random_uniform([], minval=0, maxval=3, dtype=tf.int32))

		# Apply brightness and contrast in randomized order. Uniformly sample from {0,1} and cast to tf.bool
		rand_order = tf.cast(tf.random_uniform([], minval=0, maxval=1, dtype=tf.int32), dtype=tf.bool)
		# If rand_order=0, first apply brightness and then contrast, the opposite order if rand_order=1.
		image = tf.cond(pred=rand_order, true_fn=lambda: rand_brightness_contrast(1, image),
						false_fn=lambda: rand_brightness_contrast(0, image))

	# Subtract off the mean and divide by the variance of the pixels.
	image = tf.image.per_image_standardization(image)
	return image


def parse_record(raw_record):
	"""Parse CIFAR-100 image and label from a raw record."""
	# Every record consists of a label followed by the image, with a fixed number
	# of bytes for each.
	label_bytes = _LABEL
	image_bytes = _HEIGHT * _WIDTH * _DEPTH
	record_bytes = label_bytes + image_bytes

	# Convert bytes to a vector of uint8 that is record_bytes long.
	record_vector = tf.decode_raw(raw_record, tf.uint8)

	# The first byte represents the label, which we convert from uint8 to int32
	# and then to one-hot.
	label = tf.cast(record_vector[0], tf.int32)
	label = tf.one_hot(label, NUM_CLASSES)

	# The remaining bytes after the label represent the image, which we reshape
	# from [depth * height * width] to [depth, height, width].
	depth_major = tf.reshape(record_vector[label_bytes:record_bytes], [_DEPTH, _HEIGHT, _WIDTH])

	# Convert from [depth, height, width] to [height, width, depth], and cast as
	# float32.
	image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

	return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
	"""Input_fn using the tf.data input pipeline for CIFAR-10 dataset.
	Args:
	is_training: A boolean denoting whether the input is for training.
	data_dir: The directory containing the input data.
	batch_size: The number of samples per batch.
	num_epochs: The number of epochs to repeat the dataset.
	Returns:
	A tuple of images and labels.
	"""
	dataset = record_dataset(get_filenames(is_training, data_dir))

	if is_training:
		# When choosing shuffle buffer sizes, larger sizes result in better
		# randomness, while smaller sizes have better performance. Because CIFAR-10
		# is a relatively small dataset, we choose to shuffle the full epoch.
		dataset = dataset.shuffle(buffer_size=NUM_IMAGES['train'])

	dataset = dataset.map(parse_record)
	dataset = dataset.map(lambda image, label: (preprocess_image(image, is_training), label))

	dataset = dataset.prefetch(2 * batch_size)

	# We call repeat after shuffling, rather than before, to prevent separate
	# epochs from blending together.
	if num_epochs is not None:
		dataset = dataset.repeat(num_epochs)
	else:  # Allow to repeat indefinitely if 'num_epochs' is None
		dataset = dataset.repeat()

	# Batch results by up to batch_size, and then fetch the tuple from the
	# iterator.
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_one_shot_iterator()
	images, labels = iterator.get_next()

	return images, labels

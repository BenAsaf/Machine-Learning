import tensorflow as tf
import numpy as np

# Generating random data
n_observations = 2400
DATA = np.linspace(-6, 6, n_observations)
LABELS = np.sin(DATA) + np.random.uniform(-0.5, 0.5, n_observations)
# Casting to float32
DATA = DATA.astype(np.float32, copy=False)
LABELS = LABELS.astype(np.float32, copy=False)

# Will be used for matplotlib to output the same scale
MIN_X_DATA, MAX_X_DATA = np.min(DATA), np.max(DATA)
MIN_Y_LABELS, MAX_Y_LABELS = np.min(LABELS), np.max(LABELS)

# 3 equal sizes of training,validation and test
num_of_partitions = 3  # Three divisions of the total data to: Training, Validation and Test
partitions = [int((len(DATA) / num_of_partitions) * i) for i in range(1, num_of_partitions)]
TRAIN_DATA, TRAIN_LABELS = DATA[:partitions[0]], LABELS[:partitions[0]]
VALID_DATA, VALID_LABELS = DATA[partitions[0]:partitions[1]], LABELS[partitions[0]:partitions[1]]
TEST_DATA, TEST_LABELS = DATA[partitions[1]:], LABELS[partitions[1]:]


TRAIN_DATA_SORTED = TRAIN_DATA.copy()  # Will be used to evaluate the hypothesis for plot


def get_input(data_type, batch_size, num_epochs):
	if data_type == "train":
		X = tf.constant(TRAIN_DATA)
		Y = tf.constant(TRAIN_LABELS)
	elif data_type == "test":
		X = tf.constant(TEST_DATA)
		Y = tf.constant(TEST_LABELS)
	elif data_type == "valid":
		X = tf.constant(VALID_DATA)
		Y = tf.constant(VALID_LABELS)
	else:
		raise ValueError("Unknown data type: %s. Expected one of: {train, test, valid}" % data_type)
	XS_dataset = tf.data.Dataset.from_tensors([X])
	YS_dataset = tf.data.Dataset.from_tensors([Y])

	dataset = tf.data.Dataset.zip([XS_dataset, YS_dataset])

	if data_type == "train":
		dataset = dataset.shuffle(buffer_size=TRAIN_DATA.size)

	# dataset = dataset.map(parse_record)
	# dataset = dataset.map(lambda image, label: (preprocess_image(image, is_training), label))

	dataset = dataset.prefetch(3 * batch_size)

	# We call repeat after shuffling, rather than before, to prevent separate
	# epochs from blending together.
	if num_epochs is not None:
		dataset = dataset.repeat(num_epochs)
	else:  # Allow to repeat indefinitely if 'num_epochs' is None
		dataset = dataset.repeat()

	# Batch results by up to batch_size, and then fetch the tuple from the iterator.
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_one_shot_iterator()

	xs, ys = iterator.get_next()
	return xs, ys


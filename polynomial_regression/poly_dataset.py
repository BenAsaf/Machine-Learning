import tensorflow as tf
import numpy as np

# Generating random data
n_observations = 300
DATA = np.linspace(0, 10, n_observations)
np.random.shuffle(DATA)
LABELS = np.sin(DATA) + np.random.uniform(-0.5, 0.5, n_observations)
DATA = DATA.astype(np.float64, copy=False)
LABELS = LABELS.astype(np.float64, copy=False).reshape([-1, 1])

# Will be used for matplotlib to output the same scale
MIN_X_DATA, MAX_X_DATA = np.min(DATA), np.max(DATA)
MIN_Y_LABELS, MAX_Y_LABELS = np.min(LABELS), np.max(LABELS)

# 3 equal sizes of training,validation and test
num_of_partitions = 3  # Three divisions of the total data to: Training, Validation and Test
partitions = [int((len(DATA) / num_of_partitions) * i) for i in range(1, num_of_partitions)]
TRAIN_DATA, TRAIN_LABELS = DATA[:partitions[0]], LABELS[:partitions[0]]
VALID_DATA, VALID_LABELS = DATA[partitions[0]:partitions[1]], LABELS[partitions[0]:partitions[1]]
TEST_DATA, TEST_LABELS = DATA[partitions[1]:], LABELS[partitions[1]:]


def get_input(data_type, batch_size, num_epochs):
	if data_type == "train":
		X = tf.constant(TRAIN_DATA)
		Y = tf.constant(TRAIN_LABELS)
	elif data_type == "test":
		X = tf.constant(TEST_DATA)
		Y = tf.constant(TEST_LABELS)
	elif data_type == "validation":
		X = tf.constant(VALID_DATA)
		Y = tf.constant(VALID_LABELS)
	else:
		raise ValueError("Unknown data type: %s. Expected one of: {train, test, validation}" % data_type)
	XS_dataset = tf.data.Dataset.from_tensor_slices(X)
	YS_dataset = tf.data.Dataset.from_tensor_slices(Y)

	dataset = tf.data.Dataset.zip((XS_dataset, YS_dataset))

	if data_type == "train":
		dataset = dataset.shuffle(buffer_size=TRAIN_DATA.size)

	# dataset = dataset.map(parse_record)
	# dataset = dataset.map(lambda image, label: (preprocess_image(image, is_training), label))

	dataset = dataset.prefetch(2 * batch_size)

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
	# print("wtf %s" % data_type, TRAIN_DATA[0], TRAIN_LABELS[0])
	return xs, ys


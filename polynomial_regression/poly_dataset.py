import tensorflow as tf
import numpy as np

# Generating random data
_N_OBSERVATIONS = 1200
_DATA = np.linspace(0, 6, _N_OBSERVATIONS)
np.random.shuffle(_DATA)
_LABELS = np.sin(_DATA) + np.random.uniform(-0.5, 0.5, _N_OBSERVATIONS)
_DATA = _DATA.astype(np.float64, copy=False)
_LABELS = _LABELS.astype(np.float64, copy=False).reshape([-1, 1])

# 3 equal sizes of training,validation and test
_NUM_PARTITIONS = 3  # Three divisions of the total data to: Training, Validation and Test
_PARTITIONS = [int((len(_DATA) / _NUM_PARTITIONS) * i) for i in range(1, _NUM_PARTITIONS)]
_TRAIN_DATA, _TRAIN_LABELS = _DATA[:_PARTITIONS[0]], _LABELS[:_PARTITIONS[0]]
_VALID_DATA, _VALID_LABELS = _DATA[_PARTITIONS[0]:_PARTITIONS[1]], _LABELS[_PARTITIONS[0]:_PARTITIONS[1]]
_TEST_DATA, _TEST_LABELS = _DATA[_PARTITIONS[1]:], _LABELS[_PARTITIONS[1]:]


def get_input(data_type, batch_size, num_epochs):
	if data_type == "train":
		X = tf.constant(_TRAIN_DATA)
		Y = tf.constant(_TRAIN_LABELS)
	elif data_type == "test":
		X = tf.constant(_TEST_DATA)
		Y = tf.constant(_TEST_LABELS)
	elif data_type == "validation":
		X = tf.constant(_VALID_DATA)
		Y = tf.constant(_VALID_LABELS)
	else:
		raise ValueError("Unknown data type: %s. Expected one of: {train, test, validation}" % data_type)
	XS_dataset = tf.data.Dataset.from_tensor_slices(X)
	YS_dataset = tf.data.Dataset.from_tensor_slices(Y)

	dataset = tf.data.Dataset.zip((XS_dataset, YS_dataset))

	if data_type == "train":
		dataset = dataset.shuffle(buffer_size=_TRAIN_DATA.size)

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
	return xs, ys


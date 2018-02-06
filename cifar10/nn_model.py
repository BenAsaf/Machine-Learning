import tensorflow as tf
import nn_dataset

NUM_CLASSES = nn_dataset.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = nn_dataset.NUM_IMAGES["train"]
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = nn_dataset.NUM_IMAGES["validation"]


# Constants describing the training process.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def get_model(x, bIsTraining):
	with tf.name_scope("Model"):
		nn = tf.layers.batch_normalization(x, axis=-1, training=bIsTraining)
		nn = tf.layers.conv2d(nn, filters=64, kernel_size=3, strides=(1, 1), padding="same")
		nn = tf.nn.relu(nn)

		for i in range(3):
			nn = tf.layers.batch_normalization(nn, axis=-1, training=bIsTraining)
			nn = tf.layers.conv2d(nn, filters=128, kernel_size=3, strides=(1, 1), padding="same")
			nn = tf.nn.relu(nn)

		nn = tf.layers.max_pooling2d(nn, pool_size=2, strides=(2, 2), padding="valid")

		for i in range(3):
			nn = tf.layers.batch_normalization(nn, axis=-1, training=bIsTraining)
			nn = tf.layers.conv2d(nn, filters=256, kernel_size=3, strides=(1, 1), padding="same")
			nn = tf.nn.relu(nn)

		nn = tf.layers.max_pooling2d(nn, pool_size=2, strides=(2, 2), padding="valid")

		for i in range(3):
			nn = tf.layers.batch_normalization(nn, axis=-1, training=bIsTraining)
			nn = tf.layers.conv2d(nn, filters=256, kernel_size=3, strides=(1, 1), padding="same")
			nn = tf.nn.relu(nn)

		for i in range(2):
			nn = tf.layers.batch_normalization(nn, axis=-1, training=bIsTraining)
			nn = tf.layers.conv2d(nn, filters=128, kernel_size=3, strides=(1, 1), padding="same")
			nn = tf.layers.max_pooling2d(nn, pool_size=2, strides=(2, 2), padding="valid")
			nn = tf.nn.relu(nn)

		for i in range(2):
			nn = tf.layers.batch_normalization(nn, axis=-1, training=bIsTraining)
			nn = tf.layers.conv2d(nn, filters=128, kernel_size=1, strides=(1, 1), padding="same")
			nn = tf.nn.relu(nn)

		nn = tf.layers.flatten(nn)
		nn = tf.layers.dropout(nn, rate=0.8, training=bIsTraining)
		nn = tf.layers.dense(nn, units=nn.shape[-1], activation=tf.nn.relu)
		nn = tf.layers.dropout(nn, rate=0.8, training=bIsTraining)
		nn = tf.layers.dense(nn, units=nn.shape[-1], activation=tf.nn.relu)
		nn = tf.layers.dropout(nn, rate=0.5, training=bIsTraining)

		nn = tf.layers.dense(nn, units=NUM_CLASSES)
		logits = tf.nn.softmax(nn)

		return tf.cond(bIsTraining, true_fn=lambda: nn, false_fn=lambda: logits)


def get_loss_op(labels, logits):
	loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
	return tf.reduce_mean(loss_op)


def get_accuracy_op(labels, logits):
	prediction = tf.argmax(logits, axis=1)
	true_label = tf.argmax(labels, axis=1)
	accuracy_op = tf.reduce_mean(tf.cast(tf.equal(prediction, true_label), tf.float32))
	return tf.multiply(accuracy_op, tf.constant(100.0, dtype=tf.float32))


def get_train_op(loss_op, global_step, ARGS):
	# Variables that affect learning rate.
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / ARGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,
									staircase=True)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	opt = tf.train.GradientDescentOptimizer(lr)
	with tf.control_dependencies(update_ops):
		grads = opt.compute_gradients(loss_op)
		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	return apply_gradient_op


def is_prediction_correct_op(labels, logits):
	true_label = tf.argmax(labels, axis=1)
	prediction = tf.argmax(logits, axis=1)
	return tf.reshape(tf.cast(tf.equal(true_label, prediction), dtype=tf.int16), shape=[])
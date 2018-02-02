import tensorflow as tf
# import numpy as np

INITIAL_LEARNING_RATE = 0.1
DECAY_RATE = 0.1


def get_train_op(loss_op, global_step, ARGS):
	DECAY_STEPS = ARGS.num_epochs//8  # TODO Optimize decay_steps
	learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, DECAY_STEPS, DECAY_RATE)
	opt = tf.train.AdagradDAOptimizer(learning_rate=0.1, global_step=global_step)
	grads = opt.compute_gradients(loss_op)
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	return apply_gradient_op
	# return opt.minimize(loss_op, global_step)


def get_loss_op(labels, predictions):
	return tf.losses.mean_squared_error(labels=labels, predictions=predictions)


def get_accuracy_op(Y, prediction):
	# accuracy_op = tf.reduce_sum()
	return None


def get_model(input, degree):
	# X is a vector that takes an arbitrary sized batched data: (x0, x1, x2, ... xn) and builds a matrix
	# with size: (degree+1, n), where n is the size of the data.
	# the matrix looks like:
	#     [ (x0^0, x0^1, x0^2, ..., x0^d) ]
	#     [ (x1^0, x1^1, x1^2, ..., x1^d) ]
	#     [ (x2^0, x2^1, x2^2, ..., x2^d) ]
	#                   ...
	#     [ (xn^0, xn^1, xn^2, ..., xn^d) ]
	X = []
	for d in range(degree + 1):
		X.append(tf.pow(input, d, name="X_%d" % d))
	X = tf.transpose(X)  # Optional to transpose with argument in 'matmul' in the return statement.

	# Builds weight W with shape: [polynom_degree, 1].
	# i.e given degree=1, W=X^0 + X^1
	# TODO Consider different initialization? how's xavier/glorot doing?
	weights_init = tf.random_normal([degree + 1, 1], dtype=tf.float64, stddev=0.5)
	# TODO Maybe add L2 weight decay?
	W = tf.Variable(weights_init, dtype=tf.float64, name="W")

	return tf.matmul(X, W, name="predictions")
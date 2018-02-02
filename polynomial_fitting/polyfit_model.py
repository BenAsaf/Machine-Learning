import tensorflow as tf
import numpy as np


def get_train_op(loss_op, global_step, ARGS):
	# TODO Optimize decay_steps
	learning_rate = tf.train.exponential_decay(0.1, global_step, decay_rate=0.1, decay_steps=ARGS.num_epochs//8)
	opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="train_op")
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		grads = opt.compute_gradients(loss_op)
		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	return apply_gradient_op


def get_loss_op(Y, prediction):
	return tf.squared_difference(Y, prediction, name="loss_op")


def get_accuracy_op(Y, prediction):
	# accuracy_op = tf.reduce_sum()
	return None


def get_model(X, polynom_degree):
	with tf.name_scope("Weights"):  # TODO Maybe add L2 weight decay? Consider different initialization?
		weights_init = tf.random_normal([polynom_degree + 1], dtype=tf.float32, stddev=0.5)
		W = tf.Variable(weights_init, dtype=tf.float32, name="W")
	# phiX = np.array([])
	phiX = []
	with tf.name_scope("PhiX"):
		for d in range(polynom_degree + 1):
			phiX.append(tf.pow(X, d, name="X_%d" % d))
	return tf.reduce_sum(tf.multiply(W, phiX, name="prediction"))
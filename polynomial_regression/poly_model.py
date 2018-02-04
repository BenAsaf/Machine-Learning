import tensorflow as tf

INITIAL_LEARNING_RATE = 0.1
DECAY_RATE = 0.1


def get_train_op(loss_op, global_step):
	opt = tf.train.AdagradDAOptimizer(learning_rate=0.1, global_step=global_step)
	grads = opt.compute_gradients(loss_op)
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	return apply_gradient_op


def get_loss_op(labels, predictions):
	return tf.losses.mean_squared_error(labels=labels, predictions=predictions)


def get_model(data_x, degree):
	"""
	X is a vandermonde matrix where we set our batched data 'data_x' into X.
	with size: (n, degree+1), where n is the batch size of 'data_x'.
	the matrix X looks like:
		[ (x0^0, x0^1, x0^2, ..., x0^d) ]
		[ (x1^0, x1^1, x1^2, ..., x1^d) ]
		[ (x2^0, x2^1, x2^2, ..., x2^d) ]
					...
		[ (xn^0, xn^1, xn^2, ..., xn^d) ]
	"""
	X = []
	for d in range(degree + 1):
		X.append(tf.pow(data_x, d, name="X_%d" % d))
	X = tf.transpose(X)  # Optional to transpose with argument in 'matmul' in the return statement.

	# Builds weight vector W with shape: [degree+1, 1].
	# i.e given degree=2, W = (B0, B1, B2)
	# weights_init = tf.contrib.layers.xavier_initializer()
	weights_init = tf.initializers.random_normal()
	weights_reg = tf.nn.l2_normalize
	bias = tf.initializers.zeros()

	W = tf.get_variable("W", shape=[degree+1, 1], dtype=tf.float64, initializer=weights_init, regularizer=weights_reg)
	bias = tf.get_variable("bias", shape=[1], dtype=tf.float64, initializer=bias, regularizer=None)
	# Simply: X*W+b, B0*x0^0 + B1*x1^1 + .... + Bn*xn^d + b
	return tf.nn.xw_plus_b(X, W, bias)

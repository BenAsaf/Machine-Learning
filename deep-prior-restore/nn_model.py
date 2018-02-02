import tensorflow as tf
import numpy as np


def _upsample(nn, scaling=2):
	"""
	Upsamples a tensor
	:param nn: TF tensor
	:param scaling: Scaling factor
	:return: TF Tensor
	"""
	_, height, width, _ = nn.shape
	new_shape = np.array([height*scaling, width*scaling]).astype(np.int32)
	return tf.image.resize_bilinear(nn, size=new_shape)


def _batch_normalization(nn, bIsTraining, axis=3, momentum=0.1, epsilon=1e-5):
	"""
	Normalizes features
	:param nn: TF tensor
	:param axis: What to normalize
	:param momentum: Has something to do with moving averages. Read about Batch Normalization
	:param epsilon: The value to add to the denonmiator to avoid division by zero
	:return: TF tensor
	"""
	return tf.layers.batch_normalization(nn, axis=axis, momentum=momentum, epsilon=epsilon, training=bIsTraining)


def _conv2d(nn, filters, kernel_size, strides=1, padding="valid"):
	"""
	Adds padding and convolves.
	:param nn: TF tensor
	:param filters: Number of filters
	:param kernel_size: Size of kernel
	:param strides: Strides
	:param padding: Type of padding
	:return: TF tensor
	"""
	num_pad = int((kernel_size-1)/2)
	if num_pad != 0:
		pad_param = [[0, 0], [num_pad, num_pad], [num_pad, num_pad], [0, 0]]  # Channels first
		nn = tf.pad(tensor=nn, paddings=pad_param, mode="constant")
	return tf.layers.conv2d(inputs=nn, filters=filters, kernel_size=kernel_size, padding=padding, strides=strides,
							use_bias=True)


def _helper_build_deep_prior_model(i, input_tensor, bIsTraining, num_channels_down, num_channels_up, num_channels_skip,
								   kernels_size_down, kernels_size_up, kernels_size_skip):
	skip = None
	if num_channels_skip[i] != 0:  # START 'Skip' connection. Create a connection from here
		skip = _conv2d(nn=input_tensor, filters=num_channels_skip[i], kernel_size=kernels_size_skip[i])
		skip = _batch_normalization(skip, bIsTraining)
		skip = tf.nn.leaky_relu(skip)

	nn = _conv2d(nn=input_tensor, filters=num_channels_down[i], kernel_size=kernels_size_down[i], strides=2)
	nn = _batch_normalization(nn, bIsTraining)
	nn = tf.nn.leaky_relu(nn)

	nn = _conv2d(nn=nn, filters=num_channels_down[i], kernel_size=kernels_size_down[i])
	nn = _batch_normalization(nn, bIsTraining)
	nn = tf.nn.leaky_relu(nn)

	nn = _batch_normalization(nn, bIsTraining)

	# Recursive Call: GO DEEPER! (If not the deepest..)
	if i < len(num_channels_down) - 1:
		nn = _helper_build_deep_prior_model(i + 1, nn, bIsTraining, num_channels_down, num_channels_up, num_channels_skip,
											kernels_size_down, kernels_size_up, kernels_size_skip)

	nn = _upsample(nn, scaling=2)

	nn = _conv2d(nn=nn, filters=num_channels_up[i], kernel_size=kernels_size_up[i])
	nn = _batch_normalization(nn, bIsTraining)
	nn = tf.nn.leaky_relu(nn)

	if num_channels_skip[i] != 0:  # END 'Skip' Connection. Create a connection to here.
		nn = tf.concat([nn, skip], axis=3)   # 'skip' was assigned.

	nn = _conv2d(nn=nn, filters=num_channels_up[i], kernel_size=kernels_size_up[i])
	nn = _batch_normalization(nn, bIsTraining)
	nn = tf.nn.leaky_relu(nn)

	return nn


def build_deep_prior_model(z, bIsTraining, num_output_channels,
						   num_channels_down, num_channels_up, num_channels_skip,
						   kernels_size_down, kernels_size_up, kernels_size_skip):
	"""
	Builds the Deep Prior Model. The Architecture is 'Encoder-Decoder' with Skip connections. It's an Hourglass.

	NOTE: num_channels_up/down/skip and kernels_size_down/up must be of equal length.
	:param num_output_channels: Integer - The number of channels in the output
	:param num_channels_down: List\Array - The number of filters in each convolution layer while going down into the net
	:param num_channels_up: List\Array - The number of filters in each convolution layer while going up into the net
	:param num_channels_skip: List\Array - The number of filters in each 'skip' block's convolution layers
	:param kernels_size_down: List\Array - The kernel size in each convolution layer going down
	:param kernels_size_up: List\Array - The kernel size in each convolution layer going up
	:return: Keras Model
	"""
	assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip) == \
		   len(kernels_size_down) == len(kernels_size_up) == len(kernels_size_skip)

	nn = _helper_build_deep_prior_model(0, z, bIsTraining, num_channels_down, num_channels_up, num_channels_skip, kernels_size_down,
										kernels_size_up, kernels_size_skip)

	nn = _conv2d(nn=nn, filters=num_output_channels, kernel_size=kernels_size_up[0])
	nn = tf.sigmoid(nn, name="output")
	return nn


def get_loss_op(y_true, y_pred):
	mse_loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
	return mse_loss


def get_train_op(loss_op, learning_rate, global_step):
	opt = tf.train.AdamOptimizer(beta2=0.9, learning_rate=learning_rate)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		compute_grads = opt.compute_gradients(loss_op)
		apply_gradients_op = opt.apply_gradients(compute_grads, global_step=global_step)
	return apply_gradients_op


def get_input(z, img, noise_cons, batch_size=1):
	with tf.name_scope("Input"):
		z_tf = tf.constant(z, dtype=tf.float32)
		img_tf = tf.constant(img, dtype=tf.float32)

		zs_dataset = tf.data.Dataset.from_tensors(z_tf)
		imgs_dataset = tf.data.Dataset.from_tensors(img_tf)

		zs_dataset = zs_dataset.map(lambda x: x + noise_cons*tf.random_normal(z_tf.shape, dtype=tf.float32))

		dataset = tf.data.Dataset.zip((zs_dataset, imgs_dataset))
		dataset = dataset.repeat()

		iterator = dataset.make_initializable_iterator()

		Z, IMG = iterator.get_next()
		return Z, IMG, iterator.initializer

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from dpr_model import build_deep_prior_model, get_loss_op, get_train_op, get_input
import os

log_device_placement = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if log_device_placement else '3'  # To suppress Tensorflow's messages

# Some constants:
_gGRAYSCALE_CHANNELS = 1  # Number of channels in Grayscale image
_gRGB_CHANNELS = 3  # Number of channels in RGB image

_gGRAYSCALE_NDIM = 2  # Defines the dimension of Grayscale image array
_gRGB_NDIM = 3  # Defines the dimension of RGB image array

# Error messages:
_gERR_MSG_INVALID_IMG = "Expected Grayscale or RGB with shape [H,W] or [H,W,3] respectively."  # show_image()


def _show_image(img, iteration):
	"""
	Shows an image
	:param img: Array - img to show
	:return: None
	"""
	if img.ndim == _gGRAYSCALE_NDIM:  # Grayscale ndim
		plt.imshow(img, cmap="gray", interpolation="bilinear")
	elif img.ndim == _gRGB_NDIM:  # RGB
		plt.imshow(img, interpolation="bilinear")
	else:
		raise ValueError(_gERR_MSG_INVALID_IMG)
	plt.title("Iteration: %s" % str(iteration))
	plt.show()


def _train_deep_prior(corrupted_image, print_progress_tick, show_img_output_tick):
	"""
	Trains the Deep Prior. Basically repeating the same input but adding a slight variation using normal distribution
	:param corrupted_image: Image to restore
	:return: The Restored Image
	"""
	tf.reset_default_graph()  # Make sure default graph is reset.
	SOURCE_IMAGE = corrupted_image.copy()
	corrupted_tensor = _convert_to_tensor(corrupted_image)
	# Log and debugging options:
	LOG_FREQUENCY = print_progress_tick  # Every 50 iterations print loss
	IMG_LOG_FREQUENCY = show_img_output_tick  # Every 200 iterations show output

	learning_rate = 0.001  # Learning rate
	num_epochs = _approximate_num_iter(corrupted_image)  # To try and avoid overfit, since 64x64 is really small.
	num_channels_down = [64] * 5  # Number of 'filters'\'channels' for each 'down' layer
	num_channels_up = [64] * 5  # Number of 'filters'\'channels' for each 'up' layer
	num_channels_skip = [4] * 5  # Number of 'filters'\'channels' for each 'skip' layer
	kernels_size_down = [3] * 5  # Size of kernel for each 'down' layer
	kernels_size_up = [3] * 5  # Size of kernel for each 'up' layer
	kernels_size_skip = [1] * 5  # Size of kernel for each 'skip' layer
	reg_noise_std = (1.0 / 30.0)  # A constant to multiply the Normal samples generated before adding it to 'z'

	# 'z' input parameters:
	height, width, output_channels = corrupted_image.shape  # The given corrupted image shape
	# Number of input channels. In the paper it is 32 for large RGB images but we have possibly Grayscale 64x64
	z_input_channels = 3
	z_input_shape = [1, height, width, z_input_channels]  # Shape of 'z'
	uni_min, uni_max = 0.0, 0.1  # The range from which to draw samples for 'z'

	z_np = np.random.uniform(uni_min, uni_max, z_input_shape)  # This is our static 'z' input


	####################################################################################################################
	global_step = tf.train.get_or_create_global_step()
	bIsTraining = tf.placeholder(tf.bool)  # Used in BatchNormalization layers.

	x_val, y_true, iterator_initializer = get_input(z_np, corrupted_tensor, reg_noise_std)

	y_pred = build_deep_prior_model(z=x_val, bIsTraining=bIsTraining,
									num_output_channels=output_channels,
									num_channels_down=num_channels_down,
									num_channels_up=num_channels_up,
									num_channels_skip=num_channels_skip,
									kernels_size_down=kernels_size_down,
									kernels_size_up=kernels_size_up,
									kernels_size_skip=kernels_size_skip)
	loss_op = get_loss_op(y_true, y_pred)
	train_op = get_train_op(loss_op, learning_rate, global_step)

	scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(), local_init_op=tf.group(tf.local_variables_initializer(), iterator_initializer))

	class _LoggerHook(tf.train.SessionRunHook):
		"""Logs loss and runtime."""

		def __init__(self, *args, **kwargs):
			super(*args, **kwargs)
			self._start_time = time()
			self._train_losses = np.array([], dtype=np.float32)

		def begin(self):
			print("{: ^11s} | {: ^10s} | {: ^10s} | {: ^10s}".format("Duration", "Epoch", 'Samples', 'Loss'))

		def before_run(self, run_context):
			return tf.train.SessionRunArgs(loss_op)

		def after_run(self, run_context, run_values):
			loss_value = run_values.results
			self._train_losses = np.append(self._train_losses, loss_value)

			sess = run_context.session
			step = global_step.eval(session=sess)
			if LOG_FREQUENCY and (step % LOG_FREQUENCY == 0):
				current_time = time()
				duration = current_time - self._start_time
				self._start_time = current_time
				avg_train_loss = np.mean(self._train_losses, dtype=np.float32)
				print("{:0^3.3f} sec | {: ^10d} | {: ^10d} | {: ^4.6f}".format(duration, step, step, avg_train_loss))
				self._train_losses = np.array([])
			if IMG_LOG_FREQUENCY and (step % IMG_LOG_FREQUENCY == 0):
				output = sess.run(y_pred, feed_dict={x_val: z_np, bIsTraining: False})  # Returns shape [1, H, W, C]
				output = _convert_to_image(output)
				_show_image(np.hstack((SOURCE_IMAGE, output)), step)

	with tf.train.MonitoredTrainingSession(scaffold=scaffold,
										   hooks=[
											   # tf.train.StopAtStepHook(last_step=num_epochs),
											   tf.train.NanTensorHook(loss_op),
											   _LoggerHook()
										   ]) as mon_sess:
		for _ in range(num_epochs):
			mon_sess.run(train_op, feed_dict={bIsTraining: True})
		result = mon_sess.run(y_pred, feed_dict={x_val: z_np, bIsTraining: False})
		return _convert_to_image(result)



def _enforce_div(img, divisor=32):
	"""
	Crops the image dimensions to be divisible by a number.
	Example: 65x39 would turn into: 64x32
	:param img: Grayscale or RGB array
	:return: divisor: Integer
	"""
	num_rows, num_cols = img.shape[0], img.shape[1]
	if num_rows < divisor or num_cols < divisor:
		raise ValueError("Given image's Height or Width is less than 32.")
	else:
		diff_rows, diff_cols = (np.fmod(num_rows, divisor), np.fmod(num_cols, divisor))
		if diff_rows == 0 and diff_cols == 0:
			return img
		end_row, end_col = (num_rows - diff_rows), (num_cols - diff_cols)
		return img[0:end_row, 0:end_col]


def _convert_to_tensor(img):
	if img.ndim == _gGRAYSCALE_NDIM:  # Grayscale
		height, width = img.shape
		channels = 1
		img = img.reshape([1, height, width, channels])
	else:  # RGB
		height, width, channels = img.shape
		img = img.reshape([1, height, width, channels])
	return img


def _convert_to_image(tensor):
	_, height, width, channels = tensor.shape
	if channels == _gGRAYSCALE_CHANNELS:
		return tensor.reshape([height, width])
	else:  # No need to do anything
		return tensor[0]


def _approximate_num_iter(img):
	"""
	Nothing too fancy. Based on some experimentation with different dimensions for input. This is to try and avoid
	an overfit.
	:param img: Shape of an input tensor shaped.
	:return: recommended number of iterations
	"""
	if img.ndim == _gGRAYSCALE_NDIM:
		height, width = img.shape
		channels = 1
	else:
		height, width, channels = img.shape

	if height < 32 or width < 32:
		raise ValueError("Image is too small. Expected at least Height, Width = 32, 32")

	max_dim = np.maximum(height, width)
	if max_dim <= 64:
		return 100
	elif max_dim <= 128:
		return 200
	elif max_dim <= 256:
		return 250
	elif max_dim <= 512:
		return 350
	elif max_dim <= 1024:
		return 1800
	else:
		return 1800


def deep_prior_restore(corrupted_image, print_progress_tick=50, show_img_output_tick=None):
	"""
	Uses 'Deep Prior' architecture to restore a corrupted image. It works for both Grayscale and RGB images.
	My implementation is not well optimized for small images (64x64), so if you want to see actual results, feed it with
	pictures of at least 256x256 up-to 1024x1024 (Didn't try more than 1024 :-) but it is possible).
	It doesn't work well for 64x64 since it downsamples it to 2x2 patches which is too small to have a significant effect
	leading to overfit. Overfit will basically make the network to output a copy of the corrupted image.

	So again, if you want to see actual results, feed it with bigger images, preferably RGB :-)

	It will crop the given image to be divisible by 32 both in height and width.

	The network itself is "Encoder-Decoder" with 'Skip' connections. And it needs the image height and width to be
	divisible by 32 because the networks downsamples the image by 32.


	:param corrupted_image: Array - Corrupted Grayscale or RGB image. Must be height,width of at least 32,32
	:param print_progress_tick: Every tick to show loss. Can be None or an integer >= 0. Default is 50.
	:param show_img_output_tick: Every tick to show network output. Can be None or an integer >= 0. Default is None
	:return: Array - Restored Image Grayscale or RGB, depends on the corrupted_image
	"""
	if not isinstance(corrupted_image, np.ndarray):
		corrupted_image = np.array(corrupted_image)
	if corrupted_image.ndim not in {_gGRAYSCALE_NDIM, _gRGB_NDIM}:
		raise ValueError("Unknown image. Expected Grayscale or RGB")

	corrupted_image = _enforce_div(corrupted_image, divisor=32)  # Network down-samples by 32.

	restored_img = _train_deep_prior(corrupted_image, print_progress_tick, show_img_output_tick)

	# Clip just for safety. (Last layer is 'sigmoid'; Returns values in [0,1])
	return np.clip(restored_img, 0, 1).astype(np.float64)


def main():
	corruption_func = lambda img: img + np.random.normal(0.0, 0.3, img.shape)
	import PIL.Image as Image
	inputs_path_prefix = os.path.join(".", "images")
	outputs_path_prefix = os.path.join("outputs")
	files = np.array(os.listdir(inputs_path_prefix))
	np.random.shuffle(files)
	for f in files:
		img_path = os.path.join(inputs_path_prefix, f)
		output_path = os.path.join(outputs_path_prefix, f)

		img = Image.open(img_path)
		new_shape = (img.size[0]//2, img.size[1]//2)
		img = img.resize(new_shape)

		y = np.array(img, dtype=np.float32) / 255.0
		x = corruption_func(y)
		y_tag = deep_prior_restore(x, print_progress_tick=25, show_img_output_tick=25)

		y_tag = np.uint8(y_tag*255)
		img = Image.fromarray(y_tag)
		img.save(output_path)


if __name__ == '__main__':
	main()

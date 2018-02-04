import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from poly_dataset import get_input
from poly_model import get_model, get_train_op, get_loss_op
from time import time
import argparse
import os

_base_dir = os.path.join(".", "outputs")
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--base_dir", type=str, default=_base_dir, help="Where to output stuff")
PARSER.add_argument("--model_dir", type=str, default=os.path.join(_base_dir, "model"), help="Where to save models")
PARSER.add_argument("--plots_dir", type=str, default=os.path.join(_base_dir, "plots"), help="Where to save plots")
PARSER.add_argument("--log_frequency", type=int, default=25, help="How often to print to the console")
PARSER.add_argument("--num_epochs", type=int, default=15000, help="How many epochs to train the model")
PARSER.add_argument("--batch_size", type=int, default=1, required=False, help="Size of each batch")
PARSER.add_argument("--log_device_placement", type=bool, default=False, help="Whether or not to print Tensorflow's "
																			 "device placement")
ARGS = PARSER.parse_args()
# tf.logging.set_verbosity('0' if ARGS.log_device_placement else '3')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if ARGS.log_device_placement else '3'  # To suppress Tensorflow's messages


def export_polynomial_plot(model_number, Xs, Ys, Predictions):
	X_MIN, X_MAX = Xs.min() - 1, Xs.max() + 1
	Y_MIN, Y_MAX = np.minimum(Ys.min(), Predictions.min()) - 1, np.maximum(Ys.max(), Predictions.max()) + 1
	plt.xlim([X_MIN, X_MAX])
	plt.ylim([Y_MIN, Y_MAX])
	plt.scatter(Xs, Ys, s=5, color="blue", marker=".")
	plt.plot(Xs, Predictions, c="red", ls="solid", lw=1.0)
	plt.legend(["p(x)", "Validation data"])
	plt.axhline(0, color='black')
	plt.axvline(0, color='black')
	plt.title("Degree %d" % model_number)
	plt.xlabel('x')
	plt.ylabel('y')
	OUTPUT_SAVE_PATH = os.path.join(ARGS.plots_dir, "deg_%s_fit_plot.svg" % model_number)
	plt.savefig(OUTPUT_SAVE_PATH)
	plt.close()


def train_model(degree):
	if degree <= 0:
		raise ValueError("Invalid argument. degree <= 0")
	tf.reset_default_graph()
	CURRENT_MODEL_SAVE_PATH = os.path.join(ARGS.model_dir, str(degree))

	bIsTraining = tf.placeholder(tf.bool)

	X_train, Y_train = get_input("train", ARGS.batch_size, None)
	X_validation, Y_validation = get_input("validation", ARGS.batch_size, None)
	X = tf.cond(bIsTraining, true_fn=lambda: X_train, false_fn=lambda: X_validation)
	Y = tf.cond(bIsTraining, true_fn=lambda: Y_train, false_fn=lambda: Y_validation)

	global_step = tf.train.get_or_create_global_step()
	predictions = get_model(X, degree)
	loss_op = get_loss_op(Y, predictions)
	train_op = get_train_op(loss_op, global_step)
	# total_correct_preds, accuracy_op = tf.metrics.accuracy(Y, predictions)  Loss ~= Accuracy here
	scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(), local_init_op=tf.local_variables_initializer())

	class _LoggerHook(tf.train.SessionRunHook):
		"""Logs loss and runtime."""

		def __init__(self, *args, **kwargs):
			super(*args, **kwargs)
			self._start_time = time()
			self._train_losses = np.array([], dtype=np.float64)

		def begin(self):
			print("%8s | %10s | %11s | %16s | %16s" % (
			"Duration", "Epoch", 'Samples', 'Train Loss', 'Validation Loss'))

		def before_run(self, run_context):
			# return tf.train.SessionRunArgs([loss_op, accuracy_op])
			return tf.train.SessionRunArgs(loss_op)

		def after_run(self, run_context, run_values):
			# loss_value, accuracy_value = run_values.results
			loss_value = run_values.results
			self._train_losses = np.append(self._train_losses, loss_value)
			sess = run_context.session
			step = global_step.eval(session=sess)
			if step % ARGS.log_frequency == 0:
				current_time = time()
				duration = current_time - self._start_time
				self._start_time = current_time
				avg_train_loss = np.mean(self._train_losses, dtype=np.float64)
				avg_validation_loss = loss_op.eval(session=sess, feed_dict={bIsTraining: False})
				print("%3.2f sec | %10d | %11d | %10.6f | %10.6f" % (duration, step, step * ARGS.batch_size,
																				avg_train_loss, avg_validation_loss))
				self._train_losses = np.array([])


	with tf.train.MonitoredTrainingSession(checkpoint_dir=CURRENT_MODEL_SAVE_PATH, scaffold=scaffold,
										   hooks=[tf.train.StopAtStepHook(last_step=ARGS.num_epochs),
												  tf.train.NanTensorHook(loss_op),
												  _LoggerHook()]) as mon_sess:
		while not mon_sess.should_stop():
			mon_sess.run(train_op, feed_dict={bIsTraining: True})


def test_model(degree):
	tf.reset_default_graph()
	CURRENT_MODEL_SAVE_PATH = os.path.join(ARGS.model_dir, str(degree))
	X_data, Y_data = get_input("test", batch_size=1, num_epochs=1)
	predict_op = get_model(X_data, degree)

	Xs_coordinates, Ys_coordinates = np.array([]), np.array([])
	Preds_coordinates = np.array([])

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint(CURRENT_MODEL_SAVE_PATH))
		print("Predicting for model %d" % degree)
		try:
			while True:
				x, y = sess.run([X_data, Y_data])
				preds = sess.run(predict_op, feed_dict={X_data: x, Y_data: y})
				Xs_coordinates = np.concatenate((Xs_coordinates, x))
				Ys_coordinates = np.concatenate((Ys_coordinates, y.flatten()))
				Preds_coordinates = np.concatenate((Preds_coordinates, preds.flatten()))
		except tf.errors.OutOfRangeError:  # Obtaining data until there is none left.
			pass
	print("Preparing data for plotting.")
	indices = np.argsort(Xs_coordinates)
	Xs_coordinates = Xs_coordinates[indices]
	Ys_coordinates = Ys_coordinates[indices]
	Preds_coordinates = Preds_coordinates[indices]
	print("Exporting plot for model %d" % degree)
	export_polynomial_plot(degree, Xs_coordinates, Ys_coordinates, Preds_coordinates)


def _validate_folders():
	if not os.path.exists(ARGS.base_dir):
		os.mkdir(ARGS.base_dir)
	if not os.path.exists(ARGS.model_dir):
		os.mkdir(ARGS.model_dir)
	if not os.path.exists(ARGS.plots_dir):
		os.mkdir(ARGS.plots_dir)


def main():
	_validate_folders()
	model_degrees = range(1, 10)
	for d in model_degrees:
		train_model(degree=d)
		test_model(degree=d)


if __name__ == '__main__':
	main()

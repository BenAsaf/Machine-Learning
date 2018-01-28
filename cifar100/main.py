import tensorflow as tf
from time import time
import numpy as np
import cifar100_dataset
from cifar100_model import get_model, get_accuracy_op, get_loss_op, get_train_op, is_prediction_correct_op
import os
import argparse
from sys import stdout


PARSER = argparse.ArgumentParser()

PARSER.add_argument("--model_dir", type=str, default="./model")
PARSER.add_argument("--base_dir", type=str, default=".", help="Where to output stuff")
PARSER.add_argument("--log_frequency", type=int, default=25, help="How often to print to the console")
PARSER.add_argument("--num_epochs", type=int, default=105000, help="How many epochs to train the model")
PARSER.add_argument("--batch_size", type=int, default=128, required=False, help="Size of each batch")
PARSER.add_argument("--log_device_placement", type=bool, default=False, help="Whether or not to print Tensorflow's "
																			 "device placement")
ARGS = PARSER.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if ARGS.log_device_placement else '3'  # To suppress Tensorflow's messages

NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def train_net():
	bIsTraining = tf.placeholder(tf.bool)
	global_step = tf.train.get_or_create_global_step()

	images_train, labels_train = cifar100_dataset.input_fn(True, ARGS.base_dir, ARGS.batch_size, None)
	images_valid, labels_valid = cifar100_dataset.input_fn(False, ARGS.base_dir, ARGS.batch_size, None)
	images = tf.cond(pred=bIsTraining, true_fn=lambda: images_train, false_fn=lambda: images_valid)
	labels = tf.cond(pred=bIsTraining, true_fn=lambda: labels_train, false_fn=lambda: labels_valid)

	logits = get_model(images, bIsTraining)

	loss_op = get_loss_op(labels, logits, global_step)
	train_op = get_train_op(loss_op, global_step, ARGS)
	accuracy_op = get_accuracy_op(labels, logits)

	scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),
								 local_init_op=tf.local_variables_initializer())

	class _LoggerHook(tf.train.SessionRunHook):
		"""Logs loss and runtime."""

		def __init__(self, *args, **kwargs):
			super(*args, **kwargs)
			self._start_time = time()
			self._train_losses = np.array([], dtype=np.float32)
			self._train_accuracies = np.array([], dtype=np.float32)

		def begin(self):
			print("%8s | %10s | %11s | %10s | %5s%% | %5s%%" % ("Duration", "Iteration", 'Samples', 'Loss', 'Train', 'Validation'))

		def before_run(self, run_context):
			return tf.train.SessionRunArgs([loss_op, accuracy_op])

		def after_run(self, run_context, run_values):
			loss_value, accuracy_value = run_values.results
			self._train_losses = np.append(self._train_losses, loss_value)
			self._train_accuracies = np.append(self._train_accuracies, accuracy_value)
			sess = run_context.session
			step = global_step.eval(session=sess)
			if step % ARGS.log_frequency == 0:
				current_time = time()
				duration = current_time - self._start_time
				self._start_time = current_time
				avg_train_loss = np.mean(self._train_losses, dtype=np.float32)
				avg_train_acc = np.mean(self._train_accuracies, dtype=np.float32)
				avg_validation_acc = accuracy_op.eval(session=sess, feed_dict={bIsTraining: False})
				print("%3.2f sec | %10d | %11d | %10.6f | %3.2f%% | %3.2f%%" % (duration, step, step*ARGS.batch_size,
																		  avg_train_loss, avg_train_acc,
																		  avg_validation_acc))
				self._train_losses, self._train_accuracies = np.array([]), np.array([])


	with tf.train.MonitoredTrainingSession(
			checkpoint_dir=ARGS.model_dir,
			scaffold=scaffold,
			hooks=[tf.train.StopAtStepHook(last_step=ARGS.num_epochs),
				   tf.train.NanTensorHook(loss_op),
				   _LoggerHook()]
	) as mon_sess:
		while not mon_sess.should_stop():
			mon_sess.run(train_op, feed_dict={bIsTraining: True})


def test_model():
	tf.reset_default_graph()

	NUM_VALIDATION_IMGS = cifar100_dataset.NUM_IMAGES["validation"]

	images, labels = cifar100_dataset.input_fn(is_training=False, data_dir=ARGS.base_dir, batch_size=1, num_epochs=1)
	bIsTraining = tf.placeholder(tf.bool)

	logits = get_model(images, bIsTraining)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint(ARGS.model_dir))

		total_correct_predictions = tf.Variable(0, dtype=tf.int16, trainable=False)
		is_correct_prediction = is_prediction_correct_op(labels, logits)

		predict_op = tf.assign_add(total_correct_predictions, is_correct_prediction)

		sess.run(tf.variables_initializer([total_correct_predictions]))

		for i in range(NUM_VALIDATION_IMGS):
			stdout.write("\rPredicting %d/%d" % (i+1, NUM_VALIDATION_IMGS))
			sess.run(predict_op, feed_dict={bIsTraining: False})
		stdout.write("\n")
		num_correct = sess.run(total_correct_predictions)
	stdout.write("Accuracy: %3.3f%%" % (100*num_correct/NUM_VALIDATION_IMGS))




def main():
	cifar100_dataset.maybe_download_and_extract(ARGS.base_dir)
	train_net()
	test_model()  # Test model on the entire test examples.


if __name__ == '__main__':
	main()
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from time import time
import shutil, os
from sys import stdout


numIter = 25000
batchSize = 128
validationSize = 4096
log_freq = 25
train_data, validation_data, test_data = input_data.read_data_sets("MNIST_data", one_hot=True, reshape=False,
																   validation_size=validationSize)


def build_model(images, bIsTraining):
	with tf.name_scope("Model"):
		nn = tf.layers.batch_normalization(images, training=bIsTraining)
		nn = tf.layers.conv2d(nn, filters=32, kernel_size=5, strides=(1, 1), padding="same")
		nn = tf.layers.average_pooling2d(nn, pool_size=5, strides=(1, 1), padding="valid")
		nn = tf.nn.relu(nn)

		for i in range(3):
			nn = tf.layers.average_pooling2d(nn, pool_size=5, strides=(1, 1), padding="valid")
			nn = tf.layers.batch_normalization(nn, training=bIsTraining)
			nn = tf.layers.conv2d(nn, filters=64, kernel_size=5, strides=(1, 1), padding="same")
			nn = tf.nn.relu(nn)

		nn = tf.layers.batch_normalization(nn, training=bIsTraining)
		nn = tf.layers.conv2d(nn, filters=32, kernel_size=5, strides=(1, 1), padding="same")
		nn = tf.layers.average_pooling2d(nn, pool_size=5, strides=(1, 1), padding="valid")
		nn = tf.nn.relu(nn)

		nn = tf.layers.flatten(nn)
		nn = tf.layers.dropout(nn, rate=0.8, training=bIsTraining)
		nn = tf.layers.dense(nn, units=nn.shape[-1])
		nn = tf.nn.relu(nn)
		nn = tf.layers.dropout(nn, rate=0.8, training=bIsTraining)
		nn = tf.layers.dense(nn, units=nn.shape[-1]//2)
		nn = tf.nn.relu(nn)
		nn = tf.layers.dropout(nn, rate=0.3, training=bIsTraining)

		nn = tf.layers.dense(nn, units=10)
		logits = tf.nn.softmax(nn)

		return tf.cond(bIsTraining, true_fn=lambda: nn, false_fn=lambda: logits)


def get_loss_op(labels, logits):
	return tf.reduce_mean(loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))


def get_accuracy_op(labels, logits):
	prediction = tf.argmax(logits, 1)

	correct_prediction = tf.equal(prediction, tf.argmax(labels, axis=1))
	accuray_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return tf.multiply(accuray_op, tf.constant(100.0, dtype=tf.float32))


def get_train_op(loss_op, global_step):

	lr = tf.train.exponential_decay(0.05, global_step, batchSize*350, 0.1, staircase=True)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	opt = tf.train.GradientDescentOptimizer(lr)
	with tf.control_dependencies(update_ops):
		grads = opt.compute_gradients(loss_op)
		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	return apply_gradient_op, lr


def train():

	images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
	labels = tf.placeholder(tf.float32, shape=[None, 10])
	bIsTraining = tf.placeholder(tf.bool)
	global_step = tf.train.get_or_create_global_step()

	logits = build_model(images, bIsTraining)

	loss_op = get_loss_op(labels, logits)
	train_op, LR = get_train_op(loss_op, global_step)
	accuracy_op = get_accuracy_op(labels, logits)

	scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),
								 local_init_op=tf.local_variables_initializer())

	# saver = tf.train.Saver(max_to_keep=1)

	# with tf.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	print("%10s | %10s | %12s | %8s" % ("Iteration", 'Samples', 'Train Loss', 'Accuracy'))
	# 	highest_acc = 0.0
	# 	t_losses = np.array([])
	# 	for i in range(numIter+1):
	# 		X, Y = mnist.train.next_batch(batch_size=batchSize)
	# 		_, t_loss = sess.run([train_op, loss_op], feed_dict={images: X, labels: Y, bIsTraining: True})
	# 		t_losses = np.append(t_losses, t_loss)
	# 		if i % 50 == 0:
	# 			X_valid, Y_valid = mnist.validation.next_batch(validationSize)
	# 			acc = accuracy_op.eval(feed_dict={images: X_valid, labels: Y_valid, bIsTraining: False})
	# 			print("%10d | %10d | %12.8f | %.3f%% | %1.6f" % (i, i * batchSize, np.mean(t_losses), acc, sess.run(LR)))
	# 			t_losses = np.array([])
	# 			if highest_acc < acc:
	# 				highest_acc = acc
	# 				saver.save(sess, "./model/model.chkpt", global_step=global_step)
	class _LoggerHook(tf.train.SessionRunHook):
		"""Logs loss and runtime."""

		def __init__(self, *args, **kwargs):
			super(*args, **kwargs)
			self._step = -1
			self._start_time = time()
			self._train_losses = np.array([], dtype=np.float32)
			self._train_accuracies = np.array([], dtype=np.float32)

		def begin(self):
			print("%8s | %9s | %11s | %10s | %5s%% | %10s%%" % ("Duration", "Epochs", 'Samples', 'Loss', 'Train', 'Validation'))

		def before_run(self, run_context):
			self._step += 1
			return tf.train.SessionRunArgs([loss_op, accuracy_op])

		def after_run(self, run_context, run_values):
			loss_value, accuracy_value = run_values.results
			self._train_losses = np.append(self._train_losses, loss_value)
			self._train_accuracies = np.append(self._train_accuracies, accuracy_value)
			if self._step % log_freq == 0:
				current_time = time()
				duration = current_time - self._start_time
				self._start_time = current_time

				X, Y = validation_data.next_batch(batchSize)
				avg_validation_acc = run_context.session.run(accuracy_op,
															 feed_dict={images: X, labels: Y, bIsTraining: False})
				avg_train_loss = np.mean(self._train_losses, dtype=np.float32)
				avg_train_acc = np.mean(self._train_accuracies, dtype=np.float32)
				print("%3.2f sec | %6d | %11d | %10.6f | %5.2f%% | %5.2f%%" % (duration, self._step, self._step*batchSize,
																		  avg_train_loss, avg_train_acc,
																		  avg_validation_acc))
				self._train_losses, self._train_accuracies = np.array([]), np.array([])

	with tf.train.MonitoredTrainingSession(checkpoint_dir="./model/",
			scaffold=scaffold,
			hooks=[tf.train.StopAtStepHook(last_step=numIter),
				   tf.train.NanTensorHook(loss_op),
				   _LoggerHook()]) as mon_sess:
		while not mon_sess.should_stop():
			X, Y = train_data.next_batch(batchSize)
			mon_sess.run(train_op, feed_dict={images: X, labels: Y, bIsTraining: True})

		# X, Y = mnist.train.next_batch(batch_size=1)
		# ytag = sess.run(prediction, feed_dict={x: X})
		# print(ytag, Y[0])


def load_and_test():
	tf.reset_default_graph()

	NUM_TEST_IMGS = test_data.num_examples

	images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
	labels = tf.placeholder(tf.float32, shape=[None, 10])
	bIsTraining = tf.placeholder(tf.bool)

	logits = build_model(images, bIsTraining)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint("./model/"))

		is_correct_prediction = tf.cast(tf.reshape(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1)), []),
										dtype=tf.int16)
		total_correct_predictions = tf.Variable(0, dtype=tf.int16, trainable=False)
		predict_op = tf.assign_add(total_correct_predictions, is_correct_prediction)
		sess.run(tf.variables_initializer([total_correct_predictions]))

		for i in range(NUM_TEST_IMGS):
			stdout.write("\rPredicting %d/%d" % (i+1, NUM_TEST_IMGS))
			X, Y = test_data.next_batch(1)
			sess.run(predict_op, feed_dict={images: X, labels:Y, bIsTraining: False})
		stdout.write("\n")
		num_correct = sess.run(total_correct_predictions)
	print("Accuracy: %3.3f%%" % (float(num_correct/NUM_TEST_IMGS)*100))


def main():
	# if os.path.exists("./model/"):
	# 	shutil.rmtree("./model/")
	train()
	load_and_test()


if __name__ == '__main__':
	main()
"""
Polynomial Fitting
By Ben Asaf
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import polyfit_dataset
from polyfit_model import get_model, get_train_op, get_loss_op, get_accuracy_op
from time import time
import argparse
import os

_base_dir = "."
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--base_dir", type=str, default=_base_dir, help="Where to output stuff")
PARSER.add_argument("--model_dir", type=str, default=os.path.join(_base_dir, "model"))
PARSER.add_argument("--log_frequency", type=int, default=25, help="How often to print to the console")
PARSER.add_argument("--num_epochs", type=int, default=1000, help="How many epochs to train the model")
PARSER.add_argument("--batch_size", type=int, default=32, required=False, help="Size of each batch")
PARSER.add_argument("--log_device_placement", type=bool, default=False, help="Whether or not to print Tensorflow's "
																			 "device placement")
ARGS = PARSER.parse_args()
# tf.logging.set_verbosity('0' if ARGS.log_device_placement else '3')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if ARGS.log_device_placement else '3'  # To suppress Tensorflow's messages


# Defining dirs to output to:
# gBaseDir = "poly_fit" + os.path.sep
# gTFModels_Path = gBaseDir + "Models" + os.path.sep
# gPlotsDir_Path = gBaseDir + "Plots" + os.path.sep
# gSummariesDir_Path = gBaseDir + "Summaries" + os.path.sep


def polynomial_fitting(degree):
	if degree <= 0:
		raise ValueError("Invalid argument. degree <= 0")
	tf.reset_default_graph()
	bIsTraining = tf.placeholder(tf.bool)
	global_step = tf.train.get_or_create_global_step()

	X_train, Y_train = polyfit_dataset.get_input(True, ARGS.batch_size, None)
	X_test, Y_test = polyfit_dataset.get_input(False, ARGS.batch_size, None)
	X = tf.cond(bIsTraining, true_fn=lambda: X_train, false_fn=lambda: X_test)
	Y = tf.cond(bIsTraining, true_fn=lambda: Y_train, false_fn=lambda: Y_test)

	predictions = get_model(X, degree)
	loss_op = get_loss_op(Y, predictions)
	train_op = get_train_op(loss_op, global_step, ARGS)
	total_correct_preds, accuracy_op = tf.metrics.accuracy(Y, predictions)
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
			print("%8s | %10s | %11s | %10s | %5s%% | %10s%%" % (
			"Duration", "Iteration", 'Samples', 'Loss', 'Train', 'Validation'))

		def before_run(self, run_context):
			# return tf.train.SessionRunArgs([loss_op, accuracy_op])
			return tf.train.SessionRunArgs(loss_op)

		def after_run(self, run_context, run_values):
			# loss_value, accuracy_value = run_values.results
			loss_value = run_values.results
			self._train_losses = np.append(self._train_losses, loss_value)
			# self._train_accuracies = np.append(self._train_accuracies, accuracy_value)
			sess = run_context.session
			step = global_step.eval(session=sess)
			if step % ARGS.log_frequency == 0:
				current_time = time()
				duration = current_time - self._start_time
				self._start_time = current_time
				avg_train_loss = np.mean(self._train_losses, dtype=np.float32)
				avg_train_acc = np.mean(self._train_accuracies, dtype=np.float32)
				# avg_validation_acc = accuracy_op.eval(session=sess)
				avg_validation_acc = 0
				print("%3.2f sec | %10d | %11d | %10.6f | %5.2f%% | %5.2f%%" % (duration, step, step * ARGS.batch_size,
																				avg_train_loss, avg_train_acc,
																				avg_validation_acc))
				self._train_losses, self._train_accuracies = np.array([]), np.array([])

	with tf.train.MonitoredTrainingSession(checkpoint_dir=".", scaffold=scaffold,  # TODO Update ckpt dir to each model
										   hooks=[tf.train.StopAtStepHook(ARGS.num_epochs),
												  tf.train.NanTensorHook(loss_op),
												  _LoggerHook()]) as mon_sess:
		while not mon_sess.should_stop():
			mon_sess.run(train_op, feed_dict={bIsTraining: True})

	# output = []
	# for x in gTrainingData_Sorted:
	# 	o = sess.run(prediction, feed_dict={X: x})
	# 	output.append(o)
	# save_path = gTFModels_Path + "deg" + str(degree) + os.path.sep + "deg_" + str(degree)
	# saver = tf.train.Saver()
	# saver.save(sess, save_path, global_step=gEpoches)
	# export_poly_fit_plot(degree, output)
	# tf.reset_default_graph()


# def export_poly_fit_plot(model_number, output):
# 	if not os.path.exists(gPlotsDir_Path):
# 		os.mkdir(gPlotsDir_Path)
# 	plt.xlim([gMin_X_Value_Data - 0.5, gMax_X_Value_Data + 0.5])
# 	plt.ylim([gMin_Y_Value_Labels - 0.5, gMax_Y_Value_Labels + 0.5])
# 	plt.plot(gTrainingData, gTrainingLabels, "b.")
# 	plt.plot(gTrainingData_Sorted, output, "r-")
# 	plt.legend(["Training Data", "p(x)"])
# 	plt.axhline(0, color='black')
# 	plt.axvline(0, color='black')
# 	plt.title("Degree %d" % model_number)
# 	plt.xlabel('x')
# 	plt.ylabel('y')
# 	plt.savefig(gPlotsDir_Path + "deg_" + str(model_number) + "_fit_plot" + ".svg")
# 	plt.close()
#
#
# def export_train_validation_errors(train_errors, validation_errors):
# 	if not os.path.exists(gPlotsDir_Path):
# 		os.mkdir(gPlotsDir_Path)
# 	plt.plot(gModelsDegree, train_errors, "b-")
# 	plt.plot(gModelsDegree, validation_errors, "r-")
# 	plt.legend(["Train Errors", "Validation Errors"])
# 	plt.axhline(0, color='black')
# 	plt.axvline(0, color='black')
# 	plt.title("Train and Validation Errors")
# 	plt.xlabel('h')
# 	plt.ylabel('Error')
# 	plt.savefig(gPlotsDir_Path + "TrainValidationErrors.svg")
# 	plt.close()


# def load_model(model_number):
# 	# print("*** Loading model #%d" % model_number)
# 	tf.reset_default_graph()
# 	currentModel = "deg%d" % model_number
# 	path = gTFModels_Path + currentModel + os.path.sep
# 	meta_file_path = glob.glob(path + "*.meta")[0]
# 	sess = tf.Session()
# 	# Loading the model:
# 	saver = tf.train.import_meta_graph(meta_file_path)
# 	saver.restore(sess, tf.train.latest_checkpoint(path))
# 	# Loading the Variables, Placeholders and Operators
# 	graph = tf.get_default_graph()
# 	X = graph.get_tensor_by_name("X:0")
# 	Y = graph.get_tensor_by_name("Y:0")
# 	W = graph.get_tensor_by_name("Weights/W:0")
# 	Predictor = graph.get_tensor_by_name("Predictor:0")
# 	compute_loss = graph.get_tensor_by_name("compute_loss:0")
# 	return sess, graph, X, Y, W, Predictor, compute_loss
#
#
# def validate_models():
# 	print("*** Validating models.")
# 	best_model_idx = -1
# 	best_model_loss = 100
# 	train_errors = []
# 	validation_errors = []
# 	for d in gModelsDegree:
# 		sess, graph, X, Y, W, Predictor, compute_loss = load_model(d)
# 		global_step = 0
# 		with sess:
# 			tf.summary.scalar("Validation_h%d_Loss" % d, compute_loss)
# 			summaries = tf.summary.merge_all()
# 			validation_writer = tf.summary.FileWriter(gSummariesDir_Path + "h%d_validation" % d)
# 			vLosses = []
# 			tLosses = []
# 			for (x, y) in zip(gValidationData, gValidationLabels):
# 				vLoss, summ = sess.run([compute_loss, summaries], feed_dict={X: x, Y: y})
# 				vLosses.append(vLoss)
# 				validation_writer.add_summary(summ, global_step)
# 				global_step += 1
# 			avgVLoss = np.mean(vLosses)
# 			validation_errors.append(avgVLoss)
# 			if avgVLoss < best_model_loss:
# 				best_model_idx, best_model_loss = d, avgVLoss
#
# 			for (x, y) in zip(gTrainingData, gTrainingLabels):
# 				tLoss = sess.run(compute_loss, feed_dict={X: x, Y: y})
# 				tLosses.append(tLoss)
# 			avgTLoss = np.mean(tLosses)
# 			train_errors.append(avgTLoss)
#
# 	export_train_validation_errors(train_errors, validation_errors)
# 	return best_model_idx, best_model_loss

#
# def test_best_model(index):
# 	sess, graph, X, Y, W, Predictor, compute_loss = load_model(index)
# 	losses = []
# 	with sess:
# 		for (x, y) in zip(gTestData, gTestLabels):
# 			loss = sess.run(compute_loss, feed_dict={X: x, Y: y})
# 			losses.append(loss)
# 	return np.mean(losses)
#
#
# def k_fold_cross_validation(k):
# 	print("*** K-Fold Cross validating with k=%d" % k)
# 	partitions = [int((len(gData[:200]) / k) * i) for i in range(1, k)]
# 	kfold_data = np.split(gData[:200], partitions)
# 	kfold_labels = np.split(gLabels[:200], partitions)
#
# 	best_kfold_index = -1
# 	best_kfold_avgloss = 50000  # Just a temporary.. it will be changed at the first iteration
# 	for d in gModelsDegree:
# 		sess, graph, X, Y, W, Predictor, compute_loss = load_model(d)
# 		with sess:
# 			losses = []
# 			for (batchx, batchy) in zip(kfold_data, kfold_labels):
# 				for (x, y) in zip(batchx, batchy):
# 					loss = sess.run(compute_loss, feed_dict={X: x, Y: y})
# 					losses.append(loss)
# 			avgLoss = np.mean(losses)
# 			if avgLoss < best_kfold_avgloss:
# 				best_kfold_index, best_kfold_avgloss = d, avgLoss
# 	return best_kfold_index, best_kfold_avgloss


def main():
	model_degrees = range(1, 3)
	for d in model_degrees:
		polynomial_fitting(d)

	# shutil.rmtree(gBaseDir, ignore_errors=True)
	# if not os.path.exists(gBaseDir):
	# 	os.mkdir(gBaseDir)
	# 	os.mkdir(gPlotsDir_Path)
	# 	os.mkdir(gTFModels_Path)
	# 	for d in gModelsDegree:
	# 		os.mkdir(os.path.join(gTFModels_Path, "deg%d" % d))
	# 	for d in gModelsDegree:  # d<--[15]
	# 		polynomial_fitting(d)
	#
	# best_index, best_avgloss = validate_models()
	# best_avgloss_test = test_best_model(best_index)
	# best_kfold_index, best_kfold_avgloss = k_fold_cross_validation(k=5)
	#
	# print("*** %23s: Returned h*=%d with avg loss %f" % ("Validation", best_index, best_avgloss))
	# print("*** %23s: Tested h*=%d, it has avg loss %f" % ("Best Model on Test data", best_index, best_avgloss_test))
	# print("*** %23s: Returned h*=%d with avg loss %f" % ("K-Fold Cross Validation", best_kfold_index, best_kfold_avgloss))


if __name__ == '__main__':
	main()

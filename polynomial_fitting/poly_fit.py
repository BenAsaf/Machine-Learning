"""
Polynomial Fitting
By Ben Asaf
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import glob
import os
import shutil

# Defining dirs to output to:
gBaseDir = "poly_fit" + os.path.sep
gTFModels_Path = gBaseDir + "Models" + os.path.sep
gPlotsDir_Path = gBaseDir + "Plots" + os.path.sep
gSummariesDir_Path = gBaseDir + "Summaries" + os.path.sep

# Generating random data
n_observations = 300
gData = np.linspace(-3, 3, n_observations)
np.random.shuffle(gData)
gLabels = np.sin(gData) + np.random.uniform(-0.5, 0.5, n_observations)
# Casting to float32
gData = gData.astype(np.float32, copy=False)
gLabels = gLabels.astype(np.float32, copy=False)

# Will be used for matplotlib to output the same scale
gMin_X_Value_Data,   gMax_X_Value_Data   = np.min(gData),   np.max(gData)
gMin_Y_Value_Labels, gMax_Y_Value_Labels = np.min(gLabels), np.max(gLabels)

class Data(Enum):
    TRAIN = 1,
    TEST = 2,
    VALIDATION = 3,

# 3 equal sizes of training,validation and test
num_of_partitions = 3  # Three divisions of the total data to: Training, Validation and Test
partitions = [int((len(gData)/num_of_partitions) * i) for i in range(1, num_of_partitions)]

gTrainingData,   gTrainingLabels =   gData[:partitions[0]],    gLabels[:partitions[0]]
gValidationData, gValidationLabels = gData[partitions[0]:partitions[1]], gLabels[partitions[0]:partitions[1]]
gTestData,       gTestLabels =       gData[partitions[1]:], gLabels[partitions[1]:]

gTrainingData_Sorted = np.sort(gTrainingData)  # Will be used to evaluate the hypothesis for plot

gModelsDegree = range(1, 16)  # The degrees of our all hypothesis; fit polynomials

gEpoches = 1000  # Number of iterations for training
gProgressTick = 25  # Will print progress: iteration, samples, loss on train, loss on test

gEpsilon_Acc = 0.0000001
# gEpsilon_Acc = 0



def getFIFODequeue(data_type):
    # Creating queue for xs, ys feeder:
    if data_type == Data.TRAIN:
        xs = tf.constant(gTrainingData)
        ys = tf.constant(gTrainingLabels)
    elif data_type == Data.TEST:
        xs = tf.constant(gTestData)
        ys = tf.constant(gTestLabels)
    elif data_type == Data.VALIDATION:
        xs = tf.constant(gValidationData)
        ys = tf.constant(gValidationLabels)
    else:
        print("ERROR: getFIFODequeue")
        exit(1)

    q = tf.FIFOQueue(capacity=100, dtypes=[tf.float32, tf.float32], shapes=[[], []])  # enqueue 5 batches
    # We use the "enqueue" operation so 1 element of the queue is the full batch
    enqueue_op = q.enqueue_many([xs, ys])
    numberOfThreads = 3
    qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
    tf.train.add_queue_runner(qr)
    x_val, y_val = q.dequeue()
    return x_val, y_val


def polynomial_fitting(degree):
    if degree <= 0:
        print("Bad input..")
        exit(1)
    tf.reset_default_graph()
    batchx = np.array(gTrainingData)
    batchy = np.array(gTrainingLabels)

    with tf.name_scope("Weights"):
        weights_init = tf.random_normal([degree+1], dtype=tf.float32, stddev=0.5)
        W = tf.Variable(weights_init, dtype=tf.float32, name="W")
    X = tf.placeholder(dtype=tf.float32, name="X")
    Y = tf.placeholder(dtype=tf.float32, name="Y")
    with tf.name_scope("PhiX"):
        phiX = []
        for d in range(degree+1):
            phiX.append(tf.pow(X, d, name="X_%d" % d))

    predictor = tf.reduce_sum(tf.multiply(W, phiX, name="Predictor"))

    compute_loss = tf.squared_difference(predictor, Y, name="compute_loss")
    train_step = tf.train.AdagradDAOptimizer(learning_rate=0.1, name="train_step",
                                             global_step=tf.constant(0, dtype=tf.int64)).minimize(compute_loss)

    tf.summary.scalar("Training_h%d_Loss" % degree, compute_loss)
    summaries = tf.summary.merge_all()

    prev_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("*** Started Polynomial Fitting with Degree <=%d" % degree)
        print("%10s | %10s | %12s | %12s" % ("Iteration", 'Samples', 'Train Loss', 'Test Loss'))

        train_writer = tf.summary.FileWriter(gSummariesDir_Path + "h%d_train" % degree, sess.graph)
        test_writer = tf.summary.FileWriter(gSummariesDir_Path + "h%d_test" % degree, sess.graph)

        # train_xs, train_ys = getFIFODequeue(Data.TRAIN)
        # test_xs, test_ys = getFIFODequeue(Data.TRAIN)

        train_losses, test_losses = [], []
        for i in range(gEpoches):
            for (x, y) in zip(batchx, batchy):
                _, loss, summ = sess.run([train_step, compute_loss, summaries], feed_dict={X: x, Y: y})
                train_losses.append(loss)
                train_writer.add_summary(summ, i)
            if not(i % gProgressTick):
                for (x, y) in zip(gTestData, gTestLabels):
                    loss, summ = sess.run([compute_loss, summaries], feed_dict={X: x, Y: y})
                    test_losses.append(loss)
                    test_writer.add_summary(summ, i)
                avgTrainLoss = np.mean(train_losses)
                avgTestLoss = np.mean(test_losses)
                train_losses, test_losses = [], []
                print("%10d | %10d | %12.8f | %12.8f" % (i, i * len(batchy), avgTrainLoss, avgTestLoss))
                if np.abs(prev_loss - avgTrainLoss) < gEpsilon_Acc:
                    print("INFO: Accuracy of %0.7f achieved. Stopping training" % gEpsilon_Acc)
                    break
                prev_loss = avgTrainLoss

        output = []
        for x in gTrainingData_Sorted:
            o = sess.run(predictor, feed_dict={X: x})
            output.append(o)
        save_path = gTFModels_Path + "deg" + str(degree) + os.path.sep + "deg_" + str(degree)
        saver = tf.train.Saver()
        saver.save(sess, save_path, global_step=gEpoches)
        export_poly_fit_plot(degree, output)
    tf.reset_default_graph()



def export_poly_fit_plot(model_number, output):
    if not os.path.exists(gPlotsDir_Path):
        os.mkdir(gPlotsDir_Path)
    plt.xlim([gMin_X_Value_Data - 0.5, gMax_X_Value_Data + 0.5])
    plt.ylim([gMin_Y_Value_Labels - 0.5, gMax_Y_Value_Labels + 0.5])
    plt.plot(gTrainingData, gTrainingLabels, "b.")
    plt.plot(gTrainingData_Sorted, output, "r-")
    plt.legend(["Training Data", "p(x)"])
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title("Degree %d" % model_number)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(gPlotsDir_Path + "deg_" + str(model_number) + "_fit_plot" + ".svg")
    plt.close()

def export_train_validation_errors(train_errors, validation_errors):
    if not os.path.exists(gPlotsDir_Path):
        os.mkdir(gPlotsDir_Path)
    plt.plot(gModelsDegree, train_errors, "b-")
    plt.plot(gModelsDegree, validation_errors, "r-")
    plt.legend(["Train Errors", "Validation Errors"])
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title("Train and Validation Errors")
    plt.xlabel('h')
    plt.ylabel('Error')
    plt.savefig(gPlotsDir_Path + "TrainValidationErrors.svg")
    plt.close()


def load_model(model_number):
    # print("*** Loading model #%d" % model_number)
    tf.reset_default_graph()
    currentModel = "deg%d" % model_number
    path = gTFModels_Path + currentModel + os.path.sep
    meta_file_path = glob.glob(path + "*.meta")[0]
    sess = tf.Session()
    # Loading the model:
    saver = tf.train.import_meta_graph(meta_file_path)
    saver.restore(sess, tf.train.latest_checkpoint(path))
    # Loading the Variables, Placeholders and Operators
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    W = graph.get_tensor_by_name("Weights/W:0")
    Predictor = graph.get_tensor_by_name("Predictor:0")
    compute_loss = graph.get_tensor_by_name("compute_loss:0")
    return sess, graph, X, Y, W, Predictor, compute_loss


def validate_models():
    print("*** Validating models.")
    best_model_idx = -1
    best_model_loss = 100
    train_errors = []
    validation_errors = []
    for d in gModelsDegree:
        sess, graph, X, Y, W, Predictor, compute_loss = load_model(d)
        global_step = 0
        with sess:
            tf.summary.scalar("Validation_h%d_Loss" % d, compute_loss)
            summaries = tf.summary.merge_all()
            validation_writer = tf.summary.FileWriter(gSummariesDir_Path + "h%d_validation" % d)
            vLosses = []
            tLosses = []
            for (x, y) in zip(gValidationData, gValidationLabels):
                vLoss, summ = sess.run([compute_loss, summaries], feed_dict={X: x, Y: y})
                vLosses.append(vLoss)
                validation_writer.add_summary(summ, global_step)
                global_step += 1
            avgVLoss = np.mean(vLosses)
            validation_errors.append(avgVLoss)
            if avgVLoss < best_model_loss:
                best_model_idx, best_model_loss = d, avgVLoss

            for (x, y) in zip(gTrainingData, gTrainingLabels):
                tLoss = sess.run(compute_loss, feed_dict={X: x, Y: y})
                tLosses.append(tLoss)
            avgTLoss = np.mean(tLosses)
            train_errors.append(avgTLoss)

    export_train_validation_errors(train_errors, validation_errors)
    return best_model_idx, best_model_loss


def test_best_model(index):
    sess, graph, X, Y, W, Predictor, compute_loss = load_model(index)
    losses = []
    with sess:
        for (x, y) in zip(gTestData, gTestLabels):
            loss = sess.run(compute_loss, feed_dict={X: x, Y: y})
            losses.append(loss)
    return np.mean(losses)


def k_fold_cross_validation(k):
    print("*** K-Fold Cross validating with k=%d" % k)
    partitions = [int((len(gData[:200]) / k) * i) for i in range(1, k)]
    kfold_data = np.split(gData[:200], partitions)
    kfold_labels = np.split(gLabels[:200], partitions)

    best_kfold_index = -1
    best_kfold_avgloss = 50000  # Just a temporary.. it will be changed at the first iteration
    for d in gModelsDegree:
        sess, graph, X, Y, W, Predictor, compute_loss = load_model(d)
        with sess:
            losses = []
            for (batchx, batchy) in zip(kfold_data, kfold_labels):
                for (x, y) in zip(batchx, batchy):
                    loss = sess.run(compute_loss, feed_dict={X: x, Y: y})
                    losses.append(loss)
            avgLoss = np.mean(losses)
            if avgLoss < best_kfold_avgloss:
                best_kfold_index, best_kfold_avgloss = d, avgLoss
    return best_kfold_index, best_kfold_avgloss


def main():
    shutil.rmtree(gBaseDir, ignore_errors=True)
    if not os.path.exists(gBaseDir):
        os.mkdir(gBaseDir)
        os.mkdir(gPlotsDir_Path)
        os.mkdir(gTFModels_Path)
        for d in gModelsDegree:
            os.mkdir(os.path.join(gTFModels_Path, "deg%d" % d))
        for d in gModelsDegree:  # d<--[15]
            polynomial_fitting(d)

    best_index, best_avgloss = validate_models()
    best_avgloss_test = test_best_model(best_index)
    best_kfold_index, best_kfold_avgloss = k_fold_cross_validation(k=5)

    print("*** %23s: Returned h*=%d with avg loss %f" % ("Validation", best_index, best_avgloss))
    print("*** %23s: Tested h*=%d, it has avg loss %f" % ("Best Model on Test data", best_index, best_avgloss_test))
    print("*** %23s: Returned h*=%d with avg loss %f" % ("K-Fold Cross Validation", best_kfold_index, best_kfold_avgloss))


if __name__ == '__main__':
    main()

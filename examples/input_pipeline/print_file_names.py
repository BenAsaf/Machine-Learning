import tensorflow as tf
import glob
import os

path = os.path.join(".", "dummies", "*.txt")  # Create a pattern for glob to obtain all text files
file_names = glob.glob(path)  # Get a list of all files with the pattern above

F_NAME = tf.constant(file_names)  # Will be used to represent a files name

q = tf.FIFOQueue(capacity=5, dtypes=tf.string)  # Creating the FIFO queue
enqueue_op = q.enqueue_many(F_NAME)  # T
numberOfThreads = 1  # Number of threads to do this with
qr = tf.train.QueueRunner(queue=q, enqueue_ops=[enqueue_op] * numberOfThreads)
tf.train.add_queue_runner(qr)
input = q.dequeue()  # Place holder for a single file name

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        print(sess.run(input))

    coord.request_stop()
    coord.join(threads)
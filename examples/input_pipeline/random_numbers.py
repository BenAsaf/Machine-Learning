import tensorflow as tf
import numpy as np

# Generating data:
n_observations = 300
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
xs = xs.astype(np.float32, copy=False)
ys = ys.astype(np.float32, copy=False)
print(xs.dtype)
print(ys.dtype)

# Creating queue for xs, ys feeder:
x_value = tf.constant(xs)
y_value = tf.constant(ys)

q = tf.FIFOQueue(capacity=100, dtypes=[tf.float32, tf.float32], shapes=[[],[]])  # enqueue 5 batches
# We use the "enqueue" operation so 1 element of the queue is the full batch
enqueue_op = q.enqueue_many([x_value, y_value])
numberOfThreads = 2
qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
tf.train.add_queue_runner(qr)

x_val, y_val = q.dequeue()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(300*1000):
        print(sess.run([x_val, y_val]))

    coord.request_stop()
    coord.join(threads)
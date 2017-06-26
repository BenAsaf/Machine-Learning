import tensorflow as tf
import numpy as np
import os
import cifar10.reader as Reader

gBatchSize = 256
gEpoches = 10000


def build_net(images):
    with tf.name_scope("NN_Cifar"):
        print("%10s" % "Input:", images.get_shape())
        print()
        with tf.name_scope("Reshape_Layer"):
            net = tf.reshape(images, shape=[-1, 32, 32, 3], name="Reshape")
            print("%10s" % "Reshape:", net.get_shape())
        print()
        with tf.name_scope("Conv2d_Layer_1"):
            W1 = tf.Variable(tf.random_normal([5, 5, 3, 32]), name="W1")
            net = tf.nn.conv2d(net, W1, [1, 1, 1, 1], padding="SAME", name="Conv2d_1")
            print("%10s" % "W1:", W1.get_shape())
            print("%10s" % "Conv2d:", net.get_shape())
        print()
        with tf.name_scope("MaxPool_Layer_1"):
            net = tf.nn.max_pool(net, [1, 5, 5, 1], strides=[1, 1, 1, 1], padding="VALID")
            print("%10s" % "Maxpool:", net.get_shape())
        print()
        with tf.name_scope("Conv2d_Layer_2"):
            W2 = tf.Variable(tf.random_normal([5, 5, 32, 32]), name="W2")
            net = tf.nn.conv2d(net, W2, [1, 1, 1, 1], padding="SAME", name="Conv2d_2")
            print("%10s" % "W2:", W2.get_shape())
            print("%10s" % "Conv2d:", net.get_shape())
        print()
        with tf.name_scope("MaxPool_Layer_2"):
            net = tf.nn.max_pool(net, [1, 5, 5, 1], strides=[1, 1, 1, 1], padding="VALID")
            print("%10s" % "Maxpool:", net.get_shape())
        print()
        with tf.name_scope("Conv2d_Layer_3"):
            W3 = tf.Variable(tf.random_normal([5, 5, 32, 32]), name="W3")
            net = tf.nn.conv2d(net, W3, [1, 1, 1, 1], padding="SAME", name="Conv2d_2")
            print("%10s" % "W3:", W3.get_shape())
            print("%10s" % "Conv2d:", net.get_shape())
        print()
        with tf.name_scope("Flatten_Layer"):
            net = tf.contrib.layers.flatten(net)
            print("%10s" % "Flatten:", net.get_shape())
        embeddings = net
        predictions = tf.contrib.layers.fully_connected(net, 10, activation_fn=None, scope='fc4')
        print()
        print("%10s" % "FC:", predictions.get_shape())
    return embeddings, predictions


def train_net():
    prefix = os.path.join(".", "cifar-10-batches-bin")
    filenames = [os.path.join(prefix, "data_batch_%d.bin" % d) for d in range(1, 6)]

    file = tf.constant(filenames)

    with tf.variable_scope("Queue_FileNames"):
        q = tf.FIFOQueue(capacity=5, dtypes=tf.string)  # enqueue 5 batches
        # We use the "enqueue" operation so 1 element of the queue is the full batch
        enqueue_op = q.enqueue_many(file)
        numberOfThreads = 1
        qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
        tf.train.add_queue_runner(qr)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        result = Reader.read_cifar10(q)
        key, X, Y = result.key, result.image, result.label
        embeddings, predictions = build_net(X)

        compute_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(predictions, axis=1), tf.argmax(Y, axis=1)), tf.float32))
        compute_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=predictions))
        train_step = tf.train.AdagradDAOptimizer(0.1, global_step=tf.constant(0, dtype=tf.int64),
                                                 name="Optimizer").minimize(compute_loss)

        print("*** Started Cifar10")
        print("%10s | %10s | %15s | %5s%% | %5s%%" % ("Iteration", 'Samples', 'Loss', 'Train', 'Test'))
        sess.run(tf.global_variables_initializer())
        losses = []
        accuracies = []
        for i in range(gEpoches):
            _, loss, acc = sess.run([train_step, compute_loss, compute_accuracy])
            losses.append(loss)
            accuracies.append(acc)
            if not(i % 50):
                print("%10d | %10d | %15.6f | %5.2f%% | %5.2f%%" % (i, i * 50, np.mean(losses), 100.0 * np.mean(accuracies),
                                                                    100.0 * 0))
        coord.request_stop()
        coord.join(threads)




def main():
    train_net()



if __name__ == '__main__':
    main()
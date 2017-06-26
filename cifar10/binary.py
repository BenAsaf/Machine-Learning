import tensorflow as tf
import numpy as np
from six.moves import urllib
import os
import tarfile
import sys

import cifar10.reader as Reader

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'



def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = "."
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def attempt3():
    prefix = os.path.join(".", "cifar-10-batches-bin")
    filenames = [os.path.join(prefix, "data_batch_%d.bin" % d) for d in range(1, 6)]

    file = tf.constant(filenames)

    with tf.variable_scope("Queue_FileNames"):
        q = tf.FIFOQueue(capacity=5, dtypes=tf.string)  # enqueue 5 batches
        # We use the "enqueue" operation so 1 element of the queue is the full batch
        enqueue_op = q.enqueue_many(file)
        numberOfThreads = 6
        qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
        tf.train.add_queue_runner(qr)
        input = q.dequeue()  # It replaces our input placeholder

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            print(sess.run(input))

        coord.request_stop()
        coord.join(threads)


def attempt2():
    prefix = os.path.join(".", "cifar-10-batches-bin")
    fileNames = [os.path.join(prefix, "data_batch_%d.bin" % d) for d in range(1, 6)]

    filename_queue = tf.train.string_input_producer(fileNames)

    images, labels = Reader.inputs(None, prefix, 1)

    # with tf.variable_scope("Queue_Images"):
    #     q = tf.FIFOQueue(capacity=500, dtypes=tf.string)  # enqueue 5 batches
    #     # We use the "enqueue" operation so 1 element of the queue is the full batch
    #     enqueue_op = q.enqueue(fileNames)
    #     numberOfThreads = 6
    #     qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
    #     tf.train.add_queue_runner(qr)
    #     input = q.dequeue()  # It replaces our input placeholder
    #     input = tf.Print(input, data=[q.size(), input], message="Nb elements left, input:")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        print(images, labels)
        print(images, labels)
        print(images, labels)
        print(images, labels)

        # result = Reader.read_cifar10(q)
        # for i in range(1000000):
        #     key, label, image = sess.run([result.key, result.label, result.uint8image])
        #     if i == 50:
        #         tf.summary.image("Blarb", image)
        #     print("Key", "(%d)" % len(key), key)
        #     print("Label", "(%d)" % len(label), label)
        #     # print("Image", "(%d)" % len(image), image)
        #     print("Image", np.shape(image))
        # coord.request_stop()
        # coord.join(threads)

def attempt1():
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
        for i in range(500000):
            print(sess.run(result.key))

        coord.request_stop()
        coord.join(threads)





def main():
    maybe_download_and_extract()
    attempt1()



if __name__ == '__main__':
    main()
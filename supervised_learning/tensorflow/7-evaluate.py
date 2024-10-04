#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Args:
    X: input data to evaluate
    Y: one-hot labels for X
    save_path:location to load the model from

    Returns:
    the network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        graph = tf.get_default_graph()

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        predictions = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy_val = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss_val = sess.run(loss, feed_dict={x: X, y: Y})

    return predictions, accuracy_val, loss_val

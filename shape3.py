"""A deep shape classifier using convolutional layers adapted from the mnist classifier from tensorflow
    The test accuracy is about 0.97

"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import pandas as pd
import datetime


import tensorflow as tf

FLAGS = None
num_of_steps = 6000
batch_size = 50


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying shapes.

    Args:
      x: an input tensor with the dimensions (N_examples, 1024), where 1024 is the
      number of pixels in a standard shape image.

    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 3), with values
      equal to the logits of classifying the digit into one of 3 classes (0=rectangle, 1=ellipse and 2=triangle).
      keep_prob is a scalar placeholder for the probability of dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, 32, 32, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 32x32 image
    # is down to 8x8x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    # # Experiment with adding another fully connected layer 2, result shows adding another fully connected layer does not improve the accuary on predictions
    # W_fc2 = weight_variable([1024, 512])
    # b_fc2 = bias_variable([512])
    # h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 3])
    b_fc2 = bias_variable([3])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Import data
    shape_set = shape_set_input()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 1024])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 3])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        tf.summary.scalar('cross_entropy', cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)




    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                         sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
        sess.run(tf.global_variables_initializer())
        for i in range(num_of_steps):
            batch = shape_set.next_batch_train(batch_size)

            if i % 100 == 0:
                summary, train_accuracy = sess.run([merged, accuracy], feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                train_writer.add_summary(summary, i)
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: shape_set.test[0], y_: shape_set.test[1], keep_prob: 1.0}))

#Convert the images to greyscale images as the shape is of one color and the background is of another
def toGreyscale(x):
    b = x != x[0]
    c = x == x[0]
    x[b] = 0.5
    x[c] = -0.5

class shape_set_input(object):
    """A class to parse the dataset"""

    def __init__(self):
        self.file_names = ["shapeset2_1cspo_2_3.10000.train.amat", "shapeset2_1cspo_2_3.5000.valid.amat", "shapeset2_1cspo_2_3.5000.test.amat"]
        self.train = None
        self.valid = None
        self.test = None
        self.read_input()
        self.index = 0


    def read_input(self):
        for file_name in self.file_names:
            #Read the input amat file and
            data_df = pd.read_table(file_name, sep=' ', header=None, skiprows = 1)
            with open(file_name) as f:
                info = f.readline() \
                    #.replace("#size: ", "")
            print(info)
            print("Shape of the original dataframe: " + str(data_df.shape))


            #The first 1024 represents the gray tone of the pixel between 0 and 1 ( 32 * 32 )
            data_pixels_df_original = data_df.iloc[:, 0:1024].as_matrix()
            print(data_pixels_df_original.shape)
            np.apply_along_axis(toGreyscale, 1, data_pixels_df_original)


            #The 1025 number represents the shape: 0=rectangle, 1=ellipse and 2=triangle
            shape_type_df_original = data_df.iloc[:, 1024]

            if "train" in file_name:

                data_pixels_df_original_roll_left = np.roll(data_pixels_df_original.reshape(10000,32,32), -3, axis=2).reshape(10000,1024)
                data_pixels_df_original_roll_right = np.roll(data_pixels_df_original.reshape(10000,32,32), 3, axis=2).reshape(10000,1024)
                data_pixels_df_original_roll_up = np.roll(data_pixels_df_original.reshape(10000,32,32), -3, axis=1).reshape(10000,1024)
                data_pixels_df_original_roll_down = np.roll(data_pixels_df_original.reshape(10000,32,32), 3, axis=1).reshape(10000,1024)

                data_pixels_df_flip1 = np.flip(data_pixels_df_original.reshape(10000, 32, 32), 1).reshape(10000, 1024)
                data_pixels_df_flip1_roll_left = np.roll(data_pixels_df_flip1.reshape(10000,32,32), -3, axis=2).reshape(10000,1024)
                data_pixels_df_flip1_roll_right = np.roll(data_pixels_df_flip1.reshape(10000,32,32), 3, axis=2).reshape(10000,1024)
                data_pixels_df_flip1_roll_up = np.roll(data_pixels_df_flip1.reshape(10000,32,32), -3, axis=1).reshape(10000,1024)
                data_pixels_df_flip1_roll_down = np.roll(data_pixels_df_flip1.reshape(10000,32,32), 3, axis=1).reshape(10000,1024)


                data_pixels_df_flip2 = np.flip(data_pixels_df_original.reshape(10000, 32, 32), 2).reshape(10000, 1024)
                data_pixels_df_flip2_roll_left = np.roll(data_pixels_df_flip2.reshape(10000,32,32), -3, axis=2).reshape(10000,1024)
                data_pixels_df_flip2_roll_right = np.roll(data_pixels_df_flip2.reshape(10000,32,32), 3, axis=2).reshape(10000,1024)
                data_pixels_df_flip2_roll_up = np.roll(data_pixels_df_flip2.reshape(10000,32,32), -3, axis=1).reshape(10000,1024)
                data_pixels_df_flip2_roll_down = np.roll(data_pixels_df_flip2.reshape(10000,32,32), 3, axis=1).reshape(10000,1024)

                data_pixels_df_flip3 = np.flip(data_pixels_df_flip1.reshape(10000, 32, 32), 2).reshape(10000, 1024)
                data_pixels_df_flip3_roll_left = np.roll(data_pixels_df_flip3.reshape(10000,32,32), -3, axis=2).reshape(10000,1024)
                data_pixels_df_flip3_roll_right = np.roll(data_pixels_df_flip3.reshape(10000,32,32), 3, axis=2).reshape(10000,1024)
                data_pixels_df_flip3_roll_up = np.roll(data_pixels_df_flip3.reshape(10000,32,32), -3, axis=1).reshape(10000,1024)
                data_pixels_df_flip3_roll_down = np.roll(data_pixels_df_flip3.reshape(10000,32,32), 3, axis=1).reshape(10000,1024)

                data_pixels_df = np.concatenate((data_pixels_df_original, data_pixels_df_flip1, data_pixels_df_flip2,  data_pixels_df_flip3,
                                                 data_pixels_df_original_roll_left, data_pixels_df_original_roll_right, data_pixels_df_original_roll_up, data_pixels_df_original_roll_down,
                                                 data_pixels_df_flip1_roll_left, data_pixels_df_flip1_roll_right, data_pixels_df_flip1_roll_up, data_pixels_df_flip1_roll_down,
                                                 data_pixels_df_flip2_roll_left, data_pixels_df_flip2_roll_right, data_pixels_df_flip2_roll_up, data_pixels_df_flip2_roll_down,
                                                 data_pixels_df_flip3_roll_left, data_pixels_df_flip3_roll_right, data_pixels_df_flip3_roll_up, data_pixels_df_flip3_roll_down), axis=0)

                shape_type_df = np.concatenate((shape_type_df_original, shape_type_df_original, shape_type_df_original, shape_type_df_original,
                                                shape_type_df_original, shape_type_df_original, shape_type_df_original, shape_type_df_original,
                                                shape_type_df_original, shape_type_df_original, shape_type_df_original, shape_type_df_original,
                                                shape_type_df_original, shape_type_df_original, shape_type_df_original, shape_type_df_original,
                                                shape_type_df_original, shape_type_df_original, shape_type_df_original, shape_type_df_original), axis=0)

                #shape_type_df = np.tile(shape_type_df_original.as_matrix().reshape(shape_type_df_original.shape[0], 1), (8, 1))
                b = np.zeros((200000, 3))
                b[np.arange(200000), shape_type_df] = 1
                self.train = [data_pixels_df, b]
            elif "valid" in file_name:
                data_pixels_df_original_roll_left = np.roll(data_pixels_df_original.reshape(5000,32,32), -3, axis=2).reshape(5000,1024)
                data_pixels_df_original_roll_right = np.roll(data_pixels_df_original.reshape(5000,32,32), 3, axis=2).reshape(5000,1024)
                data_pixels_df_original_roll_up = np.roll(data_pixels_df_original.reshape(5000,32,32), -3, axis=1).reshape(5000,1024)
                data_pixels_df_original_roll_down = np.roll(data_pixels_df_original.reshape(5000,32,32), 3, axis=1).reshape(5000,1024)

                data_pixels_df_flip1 = np.flip(data_pixels_df_original.reshape(5000, 32, 32), 1).reshape(5000, 1024)
                data_pixels_df_flip1_roll_left = np.roll(data_pixels_df_flip1.reshape(5000,32,32), -3, axis=2).reshape(5000,1024)
                data_pixels_df_flip1_roll_right = np.roll(data_pixels_df_flip1.reshape(5000,32,32), 3, axis=2).reshape(5000,1024)
                data_pixels_df_flip1_roll_up = np.roll(data_pixels_df_flip1.reshape(5000,32,32), -3, axis=1).reshape(5000,1024)
                data_pixels_df_flip1_roll_down = np.roll(data_pixels_df_flip1.reshape(5000,32,32), 3, axis=1).reshape(5000,1024)


                data_pixels_df_flip2 = np.flip(data_pixels_df_original.reshape(5000, 32, 32), 2).reshape(5000, 1024)
                data_pixels_df_flip2_roll_left = np.roll(data_pixels_df_flip2.reshape(5000,32,32), -3, axis=2).reshape(5000,1024)
                data_pixels_df_flip2_roll_right = np.roll(data_pixels_df_flip2.reshape(5000,32,32), 3, axis=2).reshape(5000,1024)
                data_pixels_df_flip2_roll_up = np.roll(data_pixels_df_flip2.reshape(5000,32,32), -3, axis=1).reshape(5000,1024)
                data_pixels_df_flip2_roll_down = np.roll(data_pixels_df_flip2.reshape(5000,32,32), 3, axis=1).reshape(5000,1024)

                data_pixels_df_flip3 = np.flip(data_pixels_df_flip1.reshape(5000, 32, 32), 2).reshape(5000, 1024)
                data_pixels_df_flip3_roll_left = np.roll(data_pixels_df_flip3.reshape(5000,32,32), -3, axis=2).reshape(5000,1024)
                data_pixels_df_flip3_roll_right = np.roll(data_pixels_df_flip3.reshape(5000,32,32), 3, axis=2).reshape(5000,1024)
                data_pixels_df_flip3_roll_up = np.roll(data_pixels_df_flip3.reshape(5000,32,32), -3, axis=1).reshape(5000,1024)
                data_pixels_df_flip3_roll_down = np.roll(data_pixels_df_flip3.reshape(5000,32,32), 3, axis=1).reshape(5000,1024)

                data_pixels_df = np.concatenate((data_pixels_df_original, data_pixels_df_flip1, data_pixels_df_flip2,  data_pixels_df_flip3,
                                             data_pixels_df_original_roll_left, data_pixels_df_original_roll_right, data_pixels_df_original_roll_up, data_pixels_df_original_roll_down,
                                             data_pixels_df_flip1_roll_left, data_pixels_df_flip1_roll_right, data_pixels_df_flip1_roll_up, data_pixels_df_flip1_roll_down,
                                             data_pixels_df_flip2_roll_left, data_pixels_df_flip2_roll_right, data_pixels_df_flip2_roll_up, data_pixels_df_flip2_roll_down,
                                             data_pixels_df_flip3_roll_left, data_pixels_df_flip3_roll_right, data_pixels_df_flip3_roll_up, data_pixels_df_flip3_roll_down), axis=0)

                shape_type_df = np.concatenate((shape_type_df_original, shape_type_df_original, shape_type_df_original, shape_type_df_original,
                                                shape_type_df_original, shape_type_df_original, shape_type_df_original, shape_type_df_original,
                                                shape_type_df_original, shape_type_df_original, shape_type_df_original, shape_type_df_original,
                                                shape_type_df_original, shape_type_df_original, shape_type_df_original, shape_type_df_original,
                                                shape_type_df_original, shape_type_df_original, shape_type_df_original, shape_type_df_original), axis=0)
            #shape_type_df = np.tile(shape_type_df_original.as_matrix().reshape(shape_type_df_original.shape[0], 1), (8, 1))
                b = np.zeros((100000, 3))
                b[np.arange(100000), shape_type_df] = 1
                self.valid = [data_pixels_df, b]
                self.train = [np.concatenate((self.train[0], self.valid[0]), 0), np.concatenate((self.train[1], self.valid[1]), 0)]
            else:
                data_pixels_df = data_pixels_df_original
                shape_type_df = shape_type_df_original
                b = np.zeros((5000, 3))
                b[np.arange(5000), shape_type_df] = 1
                self.test = [data_pixels_df, b]
            print("Shape of the pixels dataframe: " + str(data_pixels_df.shape))

            #The color of the shape: this is actually an integer between 0 and 7. Divide by 7 to get the corresponding gray tone.
            shape_color_df = data_df.iloc[:, 1025]

            #The x coordinate of the centroid of the shape, between 0 (leftmost) and 256 (rightmost).
            #And the y coordinate of the centroid of the shape, between 0 (top) and 256 (bottom).
            shape_coordinates_df = data_df.iloc[:, 1026:1028]

            #The rotation angle of the shape, between 0 (no rotation) and 256 (full circle).
            # This can probably not be learnt reliably because there the reference point is ambiguous
            # (for instance, there is currently no way to know relatively to which side the triangle was rotated).
            shape_rotation_df = data_df.iloc[:, 1028]


            #The size of the shape, between 0 (a point) and 256 (the whole area). There is a lower bound and an upper bound.
            shape_size_df = data_df.iloc[:, 1029]

            #The elongation of the shape, between 0 (at least twice as wide as tall) and 256 (at least twice as tall as wide).
            shape_elogation_df = data_df.iloc[:, 1030]

    def next_batch_train(self, num):
        batch = [self.train[0][self.index: self.index+num], self.train[1][self.index: self.index+num]]
        self.index += num
        return batch



if __name__ == '__main__':
    print("before" + str(datetime.datetime.now()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str,
                        default='/tmp/tensorflow/shapeset/logs',
                        help='Directory for storing summary data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    print("after" + str(datetime.datetime.now()))
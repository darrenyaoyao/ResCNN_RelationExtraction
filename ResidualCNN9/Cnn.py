import tensorflow as tf
import numpy as np
import math

class RECnn(object):
    """
    A CNN for relation classification.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, position_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        with tf.device('/gpu:1'):
            # Placeholders for input, output and dropout
            self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
            self.input_p1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_p1")
            self.input_p2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_p2")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            # Keeping track of l2 regularization loss (optional)
            self.l2_loss = tf.constant(0.0)


            with tf.name_scope("position-embedding"):
                W = tf.Variable(tf.random_uniform([62,5],
                                             minval=-math.sqrt(6/(3*position_size+3*embedding_size)),
                                             maxval=math.sqrt(6/(3*position_size+3*embedding_size))),
                                             name="W")
                self.input_x_p1 = tf.nn.embedding_lookup(W, self.input_p1)
                self.input_x_p2 = tf.nn.embedding_lookup(W, self.input_p2)
                self.x = tf.concat([self.input_x, self.input_x_p1, self.input_x_p2],2)
                self.embedded_chars_expanded = tf.expand_dims(self.x, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size+2*position_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    self.l2_loss += tf.nn.l2_loss(W)
                    self.l2_loss += tf.nn.l2_loss(b)
                    for i in range(4):
                        h2 = self.Cnnblock(num_filters, h, i)
                        h = h2+h
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="hidden_feature")

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            with tf.name_scope("MLP"):
                W0 = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="W0")
                b0 = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b0")
                h0 = tf.nn.relu(tf.nn.xw_plus_b(self.h_drop, W0, b0))
                self.l2_loss += tf.nn.l2_loss(W0)
                self.l2_loss += tf.nn.l2_loss(b0)
                W1 = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b1")
                self.h1 = tf.nn.relu(tf.nn.xw_plus_b(h0, W1, b1))
                self.l2_loss += tf.nn.l2_loss(W1)
                self.l2_loss += tf.nn.l2_loss(b1)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                self.l2_loss += tf.nn.l2_loss(W)
                self.l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h1, W, b, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def Cnnblock(self, num_filters, h, i):
        #W1 = tf.Variable(tf.truncated_normal([3, 1, num_filters, num_filters], stddev=0.1), name="W1")
        W1 = tf.get_variable(
            "W1_"+str(i),
            shape=[3, 1, num_filters, num_filters],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1_"+str(i))
        conv1 = tf.nn.conv2d(
            h,
            W1,
            strides=[1,1,1,1],
            padding="SAME")
        h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
        self.l2_loss += tf.nn.l2_loss(W1)
        self.l2_loss += tf.nn.l2_loss(b1)
        #W2 = tf.Variable(tf.truncated_normal([3, 1, num_filters, num_filters], stddev=0.1), name="W2")
        W2 = tf.get_variable(
            "W2_"+str(i),
            shape=[3, 1, num_filters, num_filters],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2_"+str(i))
        conv2 = tf.nn.conv2d(
            h1,
            W2,
            strides=[1,1,1,1],
            padding="SAME")
        h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
        self.l2_loss += tf.nn.l2_loss(W2)
        self.l2_loss += tf.nn.l2_loss(b2)
        return h2

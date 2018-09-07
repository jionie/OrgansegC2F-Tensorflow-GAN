"""This module provides the a softmax cross entropy loss for training FCN.

In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



def leaky_relu(x, alpha=0.01):

    return tf.maximum(x, alpha * x)

####################################################################################################
# computing the DSC together with other values based on the label and prediction volumes\
def DSC(pred, label):

    label = tf.reshape(label, [-1])
    pred  = tf.cast(pred, tf.float32)
    pred = tf.reshape(pred, [-1])
    intersect = tf.multiply(label, pred)
    intersect = tf.reduce_sum(intersect)
    union=tf.reduce_sum(tf.add(pred, label))

    return intersect, union

def DSC_loss(logits, pred, labels):

    labels = tf.cast(labels, tf.float32)
    up_DSC, down_DSC = DSC(pred, labels)

    zeros = tf.zeros_like(labels, tf.float32)
    equal = tf.equal(labels, zeros)
    labels_reverse = tf.negative(tf.cast(equal, tf.float32))
    labels_new = tf.add(labels, labels_reverse)

    epsilon = tf.constant(value=1e-5)

    logits = leaky_relu(logits)
    
    logits = tf.reshape(logits, [-1])
    labels = tf.reshape(labels, [-1])
    labels_new = tf.reshape(labels_new, [-1])

    up_loss = tf.reduce_sum(tf.multiply(logits, labels))
    down_loss = tf.reduce_sum(tf.add(tf.abs(logits),tf.abs(labels)))

    
    return up_loss, down_loss, up_DSC, down_DSC



def DSC_loss_new(logits, pred, labels):

    labels = tf.cast(labels, tf.float32)
    up_DSC, down_DSC = DSC(pred, labels)

    zeros = tf.zeros_like(labels, tf.float32)
    equal = tf.equal(labels, zeros)
    labels_reverse = tf.negative(tf.cast(equal, tf.float32))
    labels_new = tf.add(labels, labels_reverse)

    epsilon = tf.constant(value=1e-5)

    logits = tf.nn.sigmoid(logits)
    
    logits = tf.reshape(logits, [-1])
    labels = tf.reshape(labels, [-1])
    labels_new = tf.reshape(labels_new, [-1])

    up_loss = tf.reduce_sum(tf.multiply(logits, labels))
    down_loss = tf.reduce_sum(tf.add(tf.abs(logits),tf.abs(labels)))

    
    return up_loss, down_loss, up_DSC, down_DSC

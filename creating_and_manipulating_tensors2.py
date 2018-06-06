# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 16:55:34 2018

@author: watec
"""

import tensorflow as tf

with tf.Graph().as_default():
    # A scalar (0-D tensor).
    scalar = tf.zeros([])
    
    # A vector with 3 elements.
    vector = tf.zeros([3])
    
    # A matrix with 2 row and 3 columns.
    matrix = tf.zeros([2, 3])
    
    with tf.Session() as sess:
        print("scalar has shape", scalar.get_shape(), "and value:\n", scalar.eval())
        print("vector has shape", vector.get_shape(), "and value:\n", vector.eval())
        print("matrix has shape", matrix.get_shape(), "and value:\n", matrix.eval())
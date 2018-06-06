# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:13:17 2018

@author: watec
"""

import tensorflow as tf

with tf.Graph().as_default():
    # Create a six-element vector (1-D tensor).
    primes = tf.constant([2, 3, 5, 7, 11, 13])
    
    # Create a constant scalar with value 1.
    ones = tf.constant(1, dtype=tf.int32)
    
    # Add the two tensors. The resulting tensor is six-element vector.
    just_beyond_primes =  tf.add(primes, ones)
    
    with tf.Session() as sess:
        print(just_beyond_primes.eval())
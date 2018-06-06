# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:12:45 2018

@author: watec
"""

import tensorflow as tf

g =  tf.Graph()

with g.as_default():
    # Create a variable with the initial value 3.
    v =  tf.Variable([3])
    
    # Create a variable of shape [1], with a random initial value,
    # sampled from a normal distribution with mean 1 and standard dviation 0.35
    w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))
    
    with tf.Session() as sess:
        try:
            v.eval()
        except tf.errors.FailedPreconditionError as e:
            print("Caught expected error:", e)
            
with g.as_default():
    # Create a variable with the initial value 3.
    v =  tf.Variable([3])
    
    # Create a variable of shape [1], with a random initial value,
    # sampled from a normal distribution with mean 1 and standard dviation 0.35
    w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))
    with tf.Session() as sess:
        initialization = tf.global_variables_initializer()
        sess.run(initialization)
        assignment = tf.assign(v, [7])
        print(v.eval())
        sess.run(assignment)
      
        print(v.eval())
        print(w.eval())

w
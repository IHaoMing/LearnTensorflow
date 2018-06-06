# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 16:02:03 2018

@author: watec
"""

import tensorflow as tf

#create a graph
g = tf.Graph()

#Establish the graph as the "default" grap
with g.as_default():
    #Assemble a graph consisting of the following three operations
    #   *Two tf.constant operations to create the operands.
    #   *One tf.add operation to add the two operands.
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    sum = tf.add(x, y, name="x_y_sum")
    z = tf.constant(4, name="z_const")
    new_sum = tf.add(sum, z, name="x_y_z_sum")
    
    #Now create a session.
    #The session will run the default graph.
    with tf.Session() as sess:
        print(new_sum.eval())
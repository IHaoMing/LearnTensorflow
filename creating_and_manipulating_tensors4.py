# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:24:53 2018

@author: watec
"""

import tensorflow as tf

with tf.Graph().as_default():
    # Create a matrix (2-d tensor) with 3 row and 4 colums.
    x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -3]],
                    dtype=tf.int32)
    
    # Create a matrix with 4 rows and 2 colums.
    y = tf.constant([[2, 2], [3, 5], [4, 5],[1, 6]],
                    dtype=tf.int32)
    
    # Multiply x by y
    # The resulting matrix will have 3 rows and 2 colums.
    matrix_multiply_result =  tf.matmul(x, y)
    
    with tf.Session() as sess:
        print(matrix_multiply_result.eval())
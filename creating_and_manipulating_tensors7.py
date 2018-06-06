# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:46:11 2018

@author: ihaoming
"""

"""
练习 1：改变两个张量的形状，使其能够相乘。
下面两个矢量无法进行矩阵乘法运算：

a = tf.constant([5, 3, 2, 7, 1, 4])
b = tf.constant([4, 6, 3])
请改变这两个矢量的形状，使其成为可以进行矩阵乘法运算的运算数。 然后，对变形后的张量调用矩阵乘法运算。
"""

import tensorflow as tf

with tf.Graph().as_default():
    a = tf.constant([5, 3, 2, 7, 1, 4])
    b = tf.constant([4, 6, 3])
    # a : 1x6 -> 2x3
    # b : 1x3 -> 3x1
    # result: 2x1
    reshaped_a = tf.reshape(a, [2, 3])
    reshaped_b = tf.reshape(b, [3, 1])
    result =  tf.matmul(reshaped_a, reshaped_b)
    
    with tf.Session() as sess:
        print(result.eval())


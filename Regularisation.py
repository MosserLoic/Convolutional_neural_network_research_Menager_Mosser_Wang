# -*- coding: utf-8 -*-

from keras import backend as K
import tensorflow as tf

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

def l2_reg(weight_matrix):
    return 0.01*K.sqrt(K.sum(tf.pow(weight_matrix,2)))

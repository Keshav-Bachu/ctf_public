#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 22:05:56 2018

@author: keshavbachu
"""

#note on next part, implimentation of the experience replay on single cells
import numpy as np
import tensorflow as tf

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def decaying_Reward(rewardSet):
    decayedReward = np.zeros(rewardSet.shape())
    gamma = 0.99
    
    for i in range (rewardSet.shape[0]):
        continuousAdd = 0
        for j in range(rewardSet.shape[1]):
            continuousAdd = continuousAdd * gamma + rewardSet[i][j]
            decayedReward[i][j] = continuousAdd

def generatePlaceholders(trainX, trainReward, trainAction):
    dim1X = trainX.shape[1]
    dim2X = trainX.shape[2]
    dim3X = trainX.shape[3]
    
    rewardDim1 = trainReward.shape[1]
    
    actionDim1 = trainAction.shape[1]
    
    Xtrain = tf.placeholder(shape = [None, dim1X, dim2X, dim3X], dtype=tf.float32, name = 'Xtrain')
    rewardTrain = tf.placeholder(shape = [None, rewardDim1], dtype=tf.float32, name = 'rewardTrain')
    actionTrain = tf.placeholder(shape = [None, actionDim1], dtype=tf.float32, name = 'actionTrain')
    
    return Xtrain, rewardTrain, actionTrain
    
def conv_net(input_data, num_input_channels, filter_shape, num_filters, weights = None, biases = None, layerNumber = 0):
    #weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    conv_filt_shape = [filter_shape,filter_shape, num_input_channels, num_filters]
    
    if(type(weights) != np.ndarray):
        weights = create_weights(conv_filt_shape)
        biases = create_biases(num_filters)
    else:
        weights = tf.Variable(tf.convert_to_tensor(weights, dtype=tf.float32))
        biases = tf.Variable(tf.convert_to_tensor(biases, dtype=tf.float32))
        #weights = tf.get_variable(name = "weight" + str(layerNumber), initializer = weights)
        #biases = tf.get_variable(name = "bias" + str(layerNumber), initializer = biases)
    
    out_layer = tf.nn.conv2d(input=input_data, filter= weights, strides= [1, 1, 1, 1], padding='SAME')
    out_layer += biases
    out_layer = tf.nn.max_pool(value=out_layer, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')
    out_layer = tf.nn.relu(out_layer)
    
    return out_layer, weights, biases
    
def flatten(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    
    return layer
 

def fc_layer(input,num_inputs,num_outputs, use_relu = False, weights = None, biases = None, layerNumber = 0):
    
    if(type(weights) != np.ndarray):
        weights = create_weights(shape=[num_inputs, num_outputs])
        biases = create_biases(num_outputs)
    else:
        weights = tf.Variable(tf.convert_to_tensor(weights, dtype=tf.float32))
        biases = tf.Variable(tf.convert_to_tensor(biases, dtype=tf.float32))
        #weights = tf.get_variable(name = "weight" + str(layerNumber), initializer = weights)
        #biases = tf.get_variable(name = "bias" + str(layerNumber), initializer = biases)
 
    layer = tf.matmul(input, weights) + biases
    if(use_relu == True):
        layer = tf.nn.relu(layer)
        
    return layer, weights, biases

def computeCost(actionSoftmax, rewardSet, actionSet):
    """
    chosen_action = tf.arg_max(actionSoftmax, 1)
    index = tf.range(0, tf.shape(actionSoftmax)[0]) * tf.shape(actionSoftmax)[1] + actionSet
    resp_outputs = tf.gather(tf.reshape(actionSoftmax, [-1]), index)
    
    cost = tf.reduce_mean(tf.log(resp_outputs) * rewardSet)
    return cost
    """
    
    #1 cost model:
    #reward * discountFactor * predicted_value - oldValue
    #reward is [? 26]
    #discount factor = some discrete value
    #predictedValue
    
    """
    actionSoftmax = tf.argmax(actionSoftmax, 1)
    discountFactor = 1
    actionSoftmax = tf.reshape(actionSoftmax, [-1,1])
    #cross_entropy = rewardSet + tf.cast(discountFactor * actionSoftmax, tf.float32) - tf.cast(actionSet, tf.float32)
    cross_entropy = (-tf.reduce_sum(rewardSet + tf.cast(discountFactor * actionSoftmax, tf.float32) - tf.cast(actionSet, tf.float32), reduction_indices=[1]))
    cost = tf.reduce_mean(cross_entropy)
    return cost
    """
    
    #the action taken by the network
    actionMax = tf.argmax(actionSoftmax,1)
    actionMax = tf.reshape(actionMax, [-1, 1])
    actionMax = tf.cast(actionMax, tf.float32)
    
    discountFactor = tf.constant(0.1, tf.float32)
    learnedValue = actionMax * discountFactor + rewardSet
    #cross_entropy = learnedValue - actionSet
    #cost = tf.reduce_mean(cross_entropy)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = actionSoftmax,labels = actionSet)
    cost = tf.reduce_mean(cross_entropy)
    predict = tf.arg_max(actionSoftmax, 1)
    return cost, predict, predict


    """
    cost = tf.reduce_mean(actionSoftmax)
    return cost
    """
    
    """
    #simple test cost    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = actionSoftmax,labels=actionSet)
    cost = tf.reduce_mean(cross_entropy)
    return cost
    """
"""
finalLayer: final layer of the NN before calculations
h_size: "The size of the final convolutional layer before splitting it into Advantage and Value streams"
actions: Number of available actions 
"""
def expReplayHelper(finalLayer, targetQ, self_actions, h_size = 5, actions = 5, QWIN = None):
    #streamQ = tf.layers.flatten(finalLayer)
    streamQ = finalLayer
    xavier_init = tf.contrib.layers.xavier_initializer()
    
    if(type(QWIN) != np.ndarray):
        QW  = tf.Variable(xavier_init([h_size, actions]))
    else:
        #QW = tf.get_variable(name = "QW", initializer = QWIN)
        QW = tf.Variable(tf.convert_to_tensor(QWIN, dtype = tf.float32))
    Qout = tf.matmul(streamQ, QW)
    
    predict = tf.arg_max(Qout, 1)
    randomProbability = tf.random_uniform(tf.shape(predict), 0,1)
    randomDecision = tf.random_uniform(tf.shape(predict), 0,5, tf.int32)
    #finalOutput = tf.cond(randomProbability < boundsFactor, lambda: tf.identity(randomDecision), lambda: tf.identity(predict))
    finalOutput = tf.where(randomProbability < 0.3, tf.cast(randomDecision, tf.int32), tf.cast(predict, tf.int32))
 
    #targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
    #self_actions = tf.placeholder(shape=[None],dtype=tf.int32)
    actions_onehot = tf.one_hot(self_actions, actions,dtype=tf.float32)

    Q = tf.reduce_sum(tf.multiply(Qout, actions_onehot), axis=1)

    td_error = tf.square(targetQ - Q)
    loss = tf.reduce_mean(td_error)
    
    return loss, predict, finalOutput, QW
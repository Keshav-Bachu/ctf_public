#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:25:51 2018

@author: keshavbachu
"""
import FormatData
def generateAllLocations(fieldView, unitObserve):
    AllLocations = []
    for i in range(specificGame.shape[1]): #cols
        for j in range(specificGame.shape[0]): #rows
            if(specificGame[j][i] == valueFind):
                AllLocations.append([i,j])
    
    return AllLocations
                
def generateOutputs(fieldView, unitObserve, weights, biases):
    AllLocations = []
    AllObservations = []
    AllLocations = generateAllLocations(fieldView, unitObserve)
    
    for i in AllLocations:
        temp = FormatData.getObservations(fieldView, 2, i[0], i[1])
        temp = FormatData.addPadding(temp, objectpad=-1, observationSpace = 2, objectLook = 4)
        
        AllObservations.append(temp)
    AllObservations = np.asanyarray(AllObservations)
    predictions = REL.makePredictions(weights, biases)

    return predictions
        
        
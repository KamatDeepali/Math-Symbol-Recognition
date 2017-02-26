"""
File: ParsingFeatureExtractor.py
Authors: Michael Potter, Deepali Kamat
Description: A program to compute features from .inkml files
"""

import sys
import os
import shutil
import codecs
import math
import numpy as np
import scipy as sp
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from scipy.spatial import distance


def angleVal(centBox1,centBox2):
    """

    :param centBox1 centroid of the first symbol:
    :param centBox2 centroid of the second symbol:
    :return:
    """
    len = math.sqrt((centBox1[0]-centBox2[0])**2 + (centBox1[1]-centBox2[1])**2)
    theta = np.arccos((centBox2[0]-centBox1[0])/len)
    if(centBox1[1]>centBox2[1]) :
        theta = 2*math.pi - theta

    return theta

def bbox(symbol):
    """
    Define the bounding box of a symbol
    """
    minx = float('inf')
    maxx = -float('inf')
    miny = float('inf')
    maxy = -float('inf')

    for stroke in symbol:
        for point in stroke:
            if point[0] > maxx:
                maxx = point[0]
            if point[0] < minx:
                minx = point[0]
            if point[1] > maxy:
                maxy = point[1]
            if point[1] < miny:
                miny = point[1]

    return [minx, maxx, miny, maxy]

def centroid(box):
    return (float(box[0]+box[1])/2,float(box[2]+box[3])/2)



def symboldiff(bbox1, bbox2):
    """
    Computes various distances between bounding box edges. Normalizes for size of combined box
    """
    deltas = []

    #Compute the larger bounding box size for normalization
    width = max(bbox1[1], bbox2[1]) - min(bbox1[0], bbox2[0])
    height = max(bbox1[3], bbox2[3]) - min(bbox1[2], bbox2[2])


    #Compute x gap between 1 and 2
    #deltas.append(bbox1[1]-bbox2[0])
    #Compute y gap between 1 and 2
    #deltas.append(bbox1[3]-bbox2[2])

    #Compute the difference between top of bounding boxes
    deltas.append((bbox2[3]- bbox1[3])/height)
    #Compute the difference between bottom of bounding boxes
    deltas.append((bbox2[2] - bbox1[2])/height)
    #Compute the difference between left of bounding boxes
    deltas.append((bbox2[0] - bbox1[0])/width)
    #Compute the difference between right of bounding boxes
    deltas.append((bbox2[1] - bbox1[1])/width)

    return deltas


def cdiff(bbox1, bbox2):
    """
    :param bbox1 bounding box for symbol 1:
    :param bbox2 bounding box for symbol 2:
    :return:
    """
    #Compute the larger bounding box size for normalization
    width = max(bbox1[1], bbox2[1]) - min(bbox1[0], bbox2[0])
    height = max(bbox1[3], bbox2[3]) - min(bbox1[2], bbox2[2])

    deltas = []
    centBox1 = centroid(bbox1)
    centBox2 = centroid(bbox2)
    #normalized differences
    diffX = (centBox1[0]-centBox2[0])/width
    diffY = (centBox1[1]-centBox2[1])/height
    diff_angle = angleVal(centBox1,centBox2)
    deltas.append(diffX)
    deltas.append(diffY)
    deltas.append(diff_angle)

    return deltas




def visualize(strokelist):
    """
    A function to visualize a combination of strokes
    """
    for stroke in strokelist:
        x1 = [pair[0] for pair in stroke]
        y1 = [pair[1] for pair in stroke]
        plt.plot(x1, y1)
        for i in range(0, len(x1)):
            plt.plot(x1[i], y1[i], 'ro')
    plt.show()
    plt.clf()

def computefeatures(symbolstrokes1, symbolstrokes2):
    """
    A function to compute the feature representation of a relationship between two symbols
    Inputs:
        symbolstrokes1 - A list of strokes in the first symbol
                        [stroke1, stroke2, ...] = [[x,y],[x,y],...],[[x,y],[x,y],...],...]
        symbolstrokes2 - A list of strokes in the second symbol
    """

    features = []

    bbox1 = bbox(symbolstrokes1)
    bbox2 = bbox(symbolstrokes2)

    sym = symboldiff(bbox1, bbox2)
    cdif = cdiff(bbox1,bbox2)

    for i in sym:
        features.append(i)
    for i in cdif:
        features.append(i)
    return features


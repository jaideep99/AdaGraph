import pandas as pd
import numpy as numpy
import keras
import tensorflow as tf
import cv2
import os

os.chdir('C:\\Users\\jaide\\OneDrive\\Documents\\VSCODE\\openCV\\roi')

for roi in os.listdir():

    img = cv2.imread(roi,0)
    print(img.shape)
    print(type(img[0][0]))


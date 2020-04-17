import cv2
import numpy as np
import pandas as pd
from math import pi
from utils import *
from labeler import retrieve_digits

def divide(img):

  if(img.shape[0]>345 or img.shape[1]>345):
    img = cv2.resize(img,(345,345),cv2.INTER_AREA)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  canny = cv2.Canny(gray,115,255)

  cnts,hier = cv2.findContours(canny,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

  image_external = np.zeros(canny.shape, canny.dtype)
  for i in range(len(cnts)):
    if hier[0][i][0] != -1:
      cv2.drawContours(image_external, cnts, i, 255,1)

  # cv2.imshow('img_ext',image_external)
  kern = np.ones((2,2),np.uint8)
  enlarge = np.ones((3,3),np.uint8)

  tmps = image_external.copy()
  tmps = cv2.dilate(tmps,kern,iterations=1)
  tmps = cv2.erode(tmps,kern,iterations=1)

  conts,_= cv2.findContours(tmps,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  tmps =  cv2.drawContours(tmps,conts,-1,(255,0,0),-1)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
  opening = cv2.morphologyEx(tmps, cv2.MORPH_OPEN, kernel, iterations=3)

  final = cv2.Canny(opening,125,255)

  mask = np.zeros(final.shape,final.dtype)
  cnts,hiers = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for contour in cnts:
    if is_circle(contour,(mask.shape,mask.dtype)):
      cv2.drawContours(mask,[contour],-1,255,-1)

  # cv2.imshow('mask',mask)
  circles = cv2.dilate(mask,enlarge,iterations=2)
  circles = cv2.bitwise_not(circles)
  circles = cv2.add(gray,circles)
  
  edges = cv2.dilate(mask,enlarge,iterations=3)
  edges = cv2.add(edges,gray)
  _,thresh = cv2.threshold(edges,115,255,cv2.THRESH_BINARY_INV)
  
  # cv2.imshow('eds',edges)
  okernel = np.ones((2,2),np.uint8)
  edges = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,okernel)
  edges = cv2.dilate(edges,enlarge,iterations=2)
  edges = cv2.erode(edges,enlarge,iterations=1)

  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return circles,edges


import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import get_circlecontours,get_maskedcontours
import pytesseract
import os
from labeler import get_digits



img = cv2.imread('testimages/check2.jpg')
if(img.shape[0]>345 or img.shape[1]>345):
  img = cv2.resize(img,(345,345),cv2.INTER_AREA)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

circles = get_circlecontours(gray)
rois = get_maskedcontours(gray,circles)



labels = []
for roi in rois:
  x = get_digits(roi)
  labels.append(x)
  
vertices = dict(zip(range(1,len(rois)+1),labels))

print(vertices)


cv2.waitKey(0)
cv2.destroyAllWindows()

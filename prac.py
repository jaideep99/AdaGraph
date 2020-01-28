import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import get_circlecontours,get_maskedcontours
import pytesseract
import os



img = cv2.imread('testimages/check2.jpg')
if(img.shape[0]>345 or img.shape[1]>345):
  img = cv2.resize(img,(345,345),cv2.INTER_AREA)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

circles = get_circlecontours(gray)
rois = get_maskedcontours(gray,circles)

pytesseract.pytesseract.tesseract_cmd = ("C:\\Program Files\\Tesseract-OCR\\tesseract")

dir = 'C:\\Users\\jaide\\OneDrive\\Documents\\VSCODE\\openCV\\roi'
os.chdir(dir)

i=0
for roi in rois:
  image = 'roi'+str(i)+'.png'
  cv2.imwrite(image,roi)
  test = pytesseract.image_to_string(roi, config='-l eng --oem 1 --psm 11')
  print(test)
  i+=1



cv2.waitKey(0)
cv2.destroyAllWindows()

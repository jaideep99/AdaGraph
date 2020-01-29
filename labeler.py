import cv2
import numpy as np
from utils import get_circlecontours
from imutils import contours as icnts
from utils import resize
from classifier import predict_digitnn

def get_digits(img):

    c,r = img.shape

    kernel = np.ones((3,3),np.uint8)

    canny = cv2.Canny(img,125,255)
    _,thresh =cv2.threshold(img,125,255,cv2.THRESH_BINARY_INV)
    thresh = cv2.dilate(thresh,kernel,iterations=1)
    thresh = cv2.erode(thresh,kernel,iterations=1)

    contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = icnts.sort_contours(contours,method="left-to-right")[0]
    
    out = thresh.copy()
    temp = out.copy()
    out = cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)


    digits = []

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        
        if(((x,y)!=(0,0)) and ((x,y+h)!=(0,c)) and ((x+w,y)!=(r,0)) and ((x+w,y+h)!=(r,c))):
            
            x,y,w,h = x-2,y-2,w+4,h+4

            roi = temp[y:y+h,x:x+w]

            roi = resize(roi,w,h)

            cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),1)
            
            digits.append(str(predict_digitnn(roi)))

    
    number = ''.join(digits)
    return number
        

import cv2
import numpy as np
from utils import get_circlecontours
from imutils import contours as icnts
from utils import resize
from classifier import predict_digitnn

def relu(x):
    if(x>0):
        return x
    return 0
        
def retrieve_digits(img):
    kern = np.ones((2,2),np.uint8)
    _,p = cv2.threshold(img,130,255,cv2.THRESH_BINARY_INV)

    cnts,hiers = cv2.findContours(p,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    
    show = p.copy()
    digits = []
    order = []
    for i in range(len(cnts)):
        if hiers[0][i][0] != -1:

            x,y,w,h = cv2.boundingRect(cnts[i])

            if(w>10 or h>10):

                xs,ys = x,y
                x,y = relu(x-2),relu(y-2)
                w,h = relu(w+2*(xs-x)),relu(h+2*(ys-y))
                roi = show[y:y+h,x:x+w]

                roi = resize(roi,w,h)
                
                digits.append(str(predict_digitnn(roi)))
                order.append(x)


    digits = [x for _,x in sorted(zip(order,digits))]
    digits = ''.join(digits)

    return digits
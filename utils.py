import cv2
import numpy as np
import math

def get_circlecontours(gray):

    kernel = np.ones((5,5),np.uint8)

    canny = cv2.Canny(gray,115,255)
    gradient = cv2.morphologyEx(canny,cv2.MORPH_GRADIENT,kernel)
    gradient = cv2.erode(gradient,kernel,iterations=1)
    contours, hierarchy = cv2.findContours(gradient,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    output = gradient.copy()
    for contour in contours:
        cv2.fillPoly(output, pts =[contour], color=(255,255,255))
    output = cv2.Canny(output,115,255)
    contours, hierarchy = cv2.findContours(output,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 9)):
            circles.append(contour)

    
    return circles

def get_insquare(x,y,w,h):
    r = max(w,h)
    z = round(((r/2)-(r/(2*(2**0.5)))))
    w = round(r/(2**0.5))
    h = w

    x,y = x+z,y+z

    return x,y,w,h

def get_maskedcontours(gray,contours):
    out = gray.copy()
    temp = gray.copy()
    rois = []
    
    i=0
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        x,y,w,h = get_insquare(x,y,w,h)
        # x,y,w,h = x+10,y+10,w-20,h-20
        rois.append(temp[y:y+h,x:x+w])
        cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),2)
        i+=1

        
    return rois


def resize(roi,w,h):
    if(w>28 and h>28):
        rest = cv2.resize(roi,(28,28),cv2.INTER_AREA)
        return rest

    if(w<28 and h<28):
        rest = np.zeros((28,28),dtype=np.uint8)
        xpos = math.floor((28-h)/2)
        ypos = math.floor((28-w)/2)

        rest[xpos:xpos+h,ypos:ypos+w] = roi
        return rest

    if(w>28 or h<28):
        s = max(w,h)
        rest = np.zeros((s,s),dtype=np.uint8)
        xpos = math.floor((s-h)/2)
        ypos = math.floor((s-w)/2)

        rest[xpos:xpos+h,ypos:ypos+w] = roi

        rest = cv2.resize(rest,(28,28),cv2.INTER_AREA)
        return rest





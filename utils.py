import cv2
import numpy as np
import math
from math import pi

def relu(x):
    if(x>0):
        return x
    return 0

def get_circlecontours(gray):

    kernel = np.ones((5,5),np.uint8)

    canny = cv2.Canny(gray,115,255)
    
    gradient = cv2.morphologyEx(canny,cv2.MORPH_GRADIENT,kernel)
    gradient = cv2.erode(gradient,kernel,iterations=1)
    contours, hierarchy = cv2.findContours(gradient,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    output = gradient.copy()
    outs = gradient.copy()
    for contour in contours:
        cv2.fillPoly(output, pts =[contour], color=(255,255,255))
    output = cv2.Canny(output,115,255)
    contours, hierarchy = cv2.findContours(output,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for contour in contours:

        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if ((len(approx) >= 9)):
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
    centers = []
    i=0
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        centers.append([relu(x-3),relu(y-3),relu(w+6),relu(h+6)])
        rois.append(temp[y:y+h,x:x+w])
        cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),2)
        i+=1

    return rois,centers


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

    else:
        s = max(w,h)
        rest = np.zeros((s,s),dtype=np.uint8)
        xpos = math.floor((s-h)/2)
        ypos = math.floor((s-w)/2)

        rest[xpos:xpos+h,ypos:ypos+w] = roi

        rest = cv2.resize(rest,(28,28),cv2.INTER_AREA)
        return rest



def skeletonize(img):

  size = np.size(img)
  skel = np.zeros(img.shape,np.uint8)
  
  ret,img = cv2.threshold(img,127,255,0)
  element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
  done = False
  
  while(not done):

      eroded = cv2.erode(img,element)
      temp = cv2.dilate(eroded,element)
      temp = cv2.subtract(img,temp)
      skel = cv2.bitwise_or(skel,temp)
      img = eroded.copy()
  
      zeros = size - cv2.countNonZero(img)
      if zeros==size:
          done = True

  return skel


def is_circle(contour,details):
    mask = np.zeros(details[0], details[1])
    cv2.drawContours(mask,[contour],-1,255,-1)
    peri = cv2.arcLength(contour, True)
    r = int(peri/(2*pi))+2

    M = cv2.moments(contour)
    try:
        cx = int(M['m10']/M['m00'])
    except:
        cx = int(M['m10']/1)    
    try:
        cy = int(M['m01']/M['m00'])
    except:
        cy = int(M['m01']/1)
    cv2.circle(mask,(cx,cy),r,0,-1)

    x = np.nonzero(mask)
    length = len(x[0])
    if(length>5):
        return False
    else:
        return True

def padding(roi,w,h):
    s = max(w,h)
    s = s+10
    rest = np.zeros((s,s),dtype=np.uint8)
    xpos = math.floor((s-h)//2)
    ypos = math.floor((s-w)//2)



    rest[xpos:xpos+h,ypos:ypos+w] = roi



    return rest

def distance(a,b):
    res = math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    return res

def quad(point,r,h):
    i,j = point
    if((0<=i)and(i<=int(r/2))and(0<=j)and(j<=int(h/2))):
        return 2
    if((int(r/2)<=i)and(i<=int(r))and(0<=j)and(j<=int(h/2))):
        return 3
    if((0<=i)and(i<=int(r/2))and(int(h/2)<=j)and(j<=int(h))):
        return 1
    if((int(r/2)<=i)and(i<=int(r))and(int(h/2)<=j)and(j<=int(h))):
        return 4

def side_density(roi):
    r,c = roi.shape
    left = roi[:,:int(c/2)]
    right = roi[:,int(c/2):]

    lcount = len(left[left==255])
    rcount = len(right[right==255])

    if(lcount>rcount):
        return 's'
    else:
        return 'e'

def orien_density(roi):
    r,c = roi.shape
    top = roi[:int(r/2),:]
    bottom = roi[int(r/2):,:]

    tcount = len(top[top==255])
    bcount = len(bottom[bottom==255])

    if(tcount>bcount):
        return 's'
    else:
        return 'e'

def quad_density(roi,slant):
    r,c = roi.shape
    q2 = roi[:int(r/2),:int(c/2)]
    q1 = roi[:int(r/2),int(c/2):]
    q3 = roi[int(r/2):,:int(c/2)]
    q4 = roi[int(r/2):,int(c/2):]

    c1 = len(q1[q1==255])
    c2 = len(q2[q2==255])
    c3 = len(q3[q3==255])
    c4 = len(q4[q4==255])


    if(slant==1):
        if(c1>c3):
            return 's'
        else:
            return 'e'

    if(slant==-1):
        if(c2>c4):
            return 's'
        else:
            return 'e'


def is_enclosed(centers,contour):

    x,y,w,h = contour
    w,h = x+w,y+h
    for cnt in centers:
        a,b,c,d = cnt
        c,d = c+a,b+d
        if((x>=a and x<=c) and (w>=a and w<=c) and (y>=b and y<=d) and (h>=b and h<=d)):
            return True
    return False

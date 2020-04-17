import numpy as np
import cv2
import math
from utils import padding,skeletonize,distance,quad,side_density,quad_density,orien_density

img = cv2.imread('edges2.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel = np.ones((2,2),np.uint8)

cnts,hiers = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

dummy = cv2.imread('testimages/tess.PNG')

i = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)

    if(w<10 and h<10):
        continue
       
    roi = gray[y:y+h,x:x+w]
    roi = cv2.erode(roi,kernel,iterations=1)
    
    tmp = np.zeros(roi.shape,roi.dtype)

    conts,hrs = cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(tmp,conts,-1,255,-1)

    left = [0,0]
    right = [0,0]
    top = [0,0]
    bottom = [0,0]
    flag = 0
    r,c = tmp.shape
    
    for i in range(r):
        for j in range(c):
            if(tmp[i,j]==255):
                if(flag==0):
                    top = bottom = left= right = [i,j]
                    flag=1
                else:
                    if j<=left[1]:
                        left = [i,j]
                    if j>=right[1]:
                        right = [i,j]
                    if i>=bottom[0]:
                        bottom = [i,j]
                    if i<=top[0]:
                        top = [i,j]

    
    t,b,r,l = quad(top,r,c),quad(bottom,r,c),quad(right,r,c),quad(left,r,c)

    # dummy[top[0],top[1]]=[0,0,255]
    # dummy[right[0],right[1]]=[255,0,0]
    # dummy[left[0],left[1]]=[255,0,0]
    # dummy[bottom[0],bottom[1]]=[0,0,255]


    height = abs(bottom[0]-top[0])
    width = abs(right[1]-left[1])

    hor = 0

    if(width>height):
        hor = 1
    else:
        hor = 0

    slant = 0
    if((t==1 and r==1) and (l==3 and b==3)):
        slant = 1
    if((t==2 and l==2) and (r==4 and b==4)):
        slant = -1
    
    start,end = top,left

    directed = False
    arrow = 's'
    if(slant==0):
        if(hor == 1):
            start = left
            end = right
            arrow = side_density(roi)

        elif(hor==0):
            start = top
            end = bottom
            arrow = orien_density(roi)
    else:
        start = top
        end = bottom
        arrow = quad_density(roi,slant)


    if(arrow=='e' and directed==True):
        start,end = end,start


    start = start[0]+y,start[1]+x
    end = end[0]+y,end[1]+x
    dummy[start[0],start[1]]=[0,0,255]
    dummy[end[0],end[1]]=[255,0,0]



cv2.imshow('out',dummy)
cv2.waitKey(0)
cv2.destroyAllWindows()
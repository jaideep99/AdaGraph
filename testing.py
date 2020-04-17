import cv2
import numpy as np

img = cv2.imread('weed1.JPG')
mask = cv2.inRange(img, (36, 25, 25), (70, 255,255))


kernel = np.ones((4,4),np.uint8)
mask = cv2.bitwise_not(mask)
mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel=kernel,iterations=3)
mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
plants = cv2.bitwise_and(mask,img)
draw = plants.copy()
plants = cv2.cvtColor(plants,cv2.COLOR_BGR2GRAY)


conts,_= cv2.findContours(plants,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


i=0
for cont in conts:
    x,y,w,h = cv2.boundingRect(cont)
    roi = img[y:y+h,x:x+w]
    cv2.imshow('roi'+str(i),roi)
    cv2.waitKey(0)
    cv2.rectangle(draw,(x,y),(x+w,y+h),(0,0,255),1)
    i+=1

cv2.imshow('detect',draw)
# cv2.imshow('weed',plants)
# cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2
import math
from utils import padding,skeletonize,distance,quad,side_density,quad_density,orien_density,is_enclosed

def join(img,centers,directed):

    gray = img

    kernel = np.ones((2,2),np.uint8)

    cnts,hiers = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    dummy = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)

    pcenters = []
    for x,y,w,h in centers:
        pcenters.append([x+int(w/2),y+int(h/2)])

    edg_dict = {k: [] for k in range(1,len(pcenters)+1)}

    i = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        if(is_enclosed(centers,[x,y,w,h])):
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

        nd1 = []
        nd2 = []
        
        for p in pcenters:
            m = distance(p,(start[1],start[0]))
            n = distance(p,(end[1],end[0]))
            nd1.append(m)
            nd2.append(n)


        nd1,nd2 = np.array(nd1),np.array(nd2)

        
        node1 = np.argmin(nd1)+1
        node2 = np.argmin(nd2)+1


        if(directed):
            edg_dict[node2].append(node1)
        else:
            edg_dict[node2].append(node1)
            edg_dict[node1].append(node2)

    # cv2.imshow('out',dummy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return edg_dict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import get_circlecontours,get_maskedcontours
import os
from labeler import retrieve_digits
from directions import divide
from join import join

filename = input()
directed = int(input())

if(directed==0):
  directed=False
else:
  directed =True
img = cv2.imread('testimages/'+filename)
if(img.shape[0]>345 or img.shape[1]>345):
  img = cv2.resize(img,(345,345),cv2.INTER_AREA)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print("dividing edges and nodes...\n")
nodes,edges = divide(img)


# cv2.imshow('image',img)
# cv2.imshow('nodes',nodes)
# cv2.imshow('edges',edges)
# cv2.waitKey(0)

print("retrieving nodes...\n")
circles = get_circlecontours(nodes)

rois,centers = get_maskedcontours(gray,circles)

print("labelling...\n")

labels = []
for roi in rois:
  x = retrieve_digits(roi)
  labels.append(x)
  
vertices = dict(zip(range(1,len(rois)+1),labels))

print("vertices : ",vertices)

raw_adj = join(edges,centers,directed)
adj_matrix = []
for x in raw_adj:
  adj_matrix.append((int(vertices[x]),[int(vertices[y]) for y in raw_adj[x]]))

print()
print("Printing Adjacency List....\n")
adj_matrix = dict(adj_matrix)
print(adj_matrix)

cv2.waitKey(0)
cv2.destroyAllWindows()


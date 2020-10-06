# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 21:26:24 2020

@author: Md. Farhadul Islam (farhad324)
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()

data_list=[]

def draw_circle(event,x,y,flags,param):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),3,(256,0,0),-1)
        x=x/100
        y=(512-y)/100
        data_list.append((x,y))
        
cv2.namedWindow(winname='Draw')

cv2.setMouseCallback('Draw',draw_circle)

img = np.zeros((512,512,3),np.uint8)

while True:
    cv2.imshow('Draw',img)
    cv2.line(img,pt1=(0,0),pt2=(0,512),color=(0,0,256),thickness=10)
    cv2.line(img,pt1=(0,512),pt2=(512,512),color=(0,0,256),thickness=10)
    for i in range(512):
        if i%32==0:
            cv2.line(img,pt1=(i,0),pt2=(i,512),color=(0,0,256),thickness=1)
            cv2.line(img,pt1=(0,i),pt2=(512,i),color=(0,0,256),thickness=1)
    if cv2.waitKey(5) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()



data = pd.DataFrame(data_list,columns=['X','y'])

X = data['X']
y = data['y']

b = 0
w = 1.5

epochs =1000
learning_rate=0.01

for epoch in range(epochs):
    y_predicted = b+ w*X
    error=y-y_predicted
    L2 = 0.5*np.mean(error**2)
    
    gradient_b= -np.mean(error)
    b=b-learning_rate*gradient_b
    gradient_w= -np.mean(error*X)
    w=w-learning_rate*gradient_w
    


plt.scatter(X,y)
plt.plot(X,y_predicted,color='black')
plt.show()

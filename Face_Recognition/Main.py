# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 23:03:01 2019

@author: aloka
"""

import cv2

#provide path to the image 
imgPath = "image.jpg"


cascadePath = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascadePath)

img = cv2.imread(imgPath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Number of faces: {0}".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 23:03:01 2019

@author: aloka
"""

import cv2

#provide path to the image 
imgPath = "maroon.jpg"
 
cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#Can be used for face side profile. Not being used currently
cascade_side_profile=cv2.CascadeClassifier("haarcascade_profileface.xml")
cascade_eye = cv2.CascadeClassifier("haarcascade_eye.xml")
img = cv2.imread(imgPath)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cascade_face.detectMultiScale(
    gray_img,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Number of faces: {0}".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    gray_eye = gray_img[y:y+h, x:x+w]
    eye_color = img[y:y+h, x:x+w]
    eyes = cascade_eye.detectMultiScale(gray_eye)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(eye_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    
cv2.imshow("Detected Faces and eyes", img)
cv2.waitKey(0)
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
haar_cascade_eyes = cv2.CascadeClassifier('haarcascade_eye.xml')

print('Enter your name:')
name = input()
font = cv2.FONT_HERSHEY_DUPLEX

def detect_Face(gray_img, orig_img):
  face = haar_cascade_face.detectMultiScale(gray_img,1.3,5)
  for (x,y,w,h) in face:
    cv2.rectangle(orig_img, (x,y), (x+w,y+h), (255,0,0), 2)
    ROI_gray = gray_img[y:y+h,x:x+w]
    ROI_orig = orig_img[y:y+h,x:x+w]
    eye = haar_cascade_eyes.detectMultiScale(ROI_gray,1.1,3)
    for (ex,ey,ew,eh) in eye:
      cv2.rectangle(ROI_orig,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.putText(orig_img,name,(x+w//2,y), font, 2, (0,0,255)) #---write the text
      #cv2.imshow('Face having name', image)
  return orig_img

video_capture = cv2.VideoCapture(0)
while True:
  _,orig_img = video_capture.read()
  gray = cv2.cvtColor(orig_img,cv2.COLOR_BGR2GRAY)
  cardboard = detect_Face(gray,orig_img)
  cv2.imshow('Video',cardboard)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
video_capture.release()
cv2.destroyAllWindows()
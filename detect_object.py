import numpy as np
import cv2

cap = cv2.VideoCapture("test.mp4")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px n
thickness = 2

img_id = 0

def create_dataset(img,id,img_id):
    cv2.imwrite("data/pic." + str(id) + "." + str(img_id) + ".jpg",img)

  
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #detect face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        #detect eye
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
        #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.putText(img, "Face", (x,y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255,0,0), 2, cv2.LINE_AA)
        coords = [x,y,w,h]
    
        if len(coords) == 4:
            id=1
            result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
            create_dataset(result,id,img_id)
    
    cv2.imshow('img',img)
    img_id += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

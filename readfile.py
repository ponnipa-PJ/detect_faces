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

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,_ = clf.predict(gray[y:y+h,x:x+w])

        if _ <= 50:
            cv2.putText(img, "Steve Jobs", (x,y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255,0,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "Unknow", (x,y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255,0,0), 2, cv2.LINE_AA)

            _="[0%]".format(round(100 - _))
            print(str(_))
            
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    coords = [x,y,w,h]
    
    if len(coords) == 4:
        id=1
        result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
    
    cv2.imshow('img',img)
    img_id += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

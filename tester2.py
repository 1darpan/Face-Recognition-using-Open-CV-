import cv2
import os
import os.path
import numpy as np
import faceRecognition as fr

name={0:"Darpan",1:"Kangana",2:"Amol",3:"Dipen",4:"Akshay"}

def trainModel():
   test = os.path.exists('trainingData.yml')
   if(not test):
       faces,faceID=fr.labels_for_training_data('trainingImages')
       face_recognizer=fr.train_classifier(faces,faceID)
       face_recognizer.write('trainingData.yml')
   else:
        face_recognizer=cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read('trainingData.yml')
   return face_recognizer

face_recognizer = trainModel()

def identify(imagePath):
    dict={}
    test_img = cv2.imread(imagePath)
    faces_detected,gray_img=fr.faceDetection(test_img)
    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        dict['Confidence']=confidence
        dict['Label']=name[label]
        print(dict)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        if(confidence>50):#If confidence more than 37 then don't print predicted face text on screen
            continue
        fr.put_text(test_img,predicted_name,x,y)

    resized_img=cv2.resize(test_img,(1000,1000))
    cv2.imshow("Face detection",resized_img)
    cv2.waitKey(0)#Waits indefinitely until a key is pressed
    cv2.destroyAllWindows

identify('TestImages/Darpan.jpg')






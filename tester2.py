import cv2
import os
import os.path
import numpy as np
import faceRecognition as fr
l2=[]
for path,subdirnames,filenames in os.walk('C:\\Users\darpa\Documents\GitHub\Face-Recognition-using-Open-CV-\\trainingImages'):
        for filename in filenames:
            id=os.path.basename(path)#fetching subdirectory names
            img_path=os.path.join(path,filename)#fetching image path

            dict={'identifier':id,'image':img_path}
            l2.append(dict)
        print(l2)

path_of_model = 'C:\\Users\darpa\Documents\GitHub\Face-Recognition-using-Open-CV-/trainingData.yml'

l1 = [{0:'Darpan','image':'C:\\Users\darpa\Documents\GitHub\Face-Recognition-using-Open-CV-\trainingImages\0'},{1:'Dipen','image':'C:\\Users\darpa\Documents\GitHub\Face-Recognition-using-Open-CV-\trainingImages\1'},{2:'Amol','image':'C:\\Users\darpa\Documents\GitHub\Face-Recognition-using-Open-CV-\trainingImages\2'},{4:'Akshay','image':'C:\\Users\darpa\Documents\GitHub\Face-Recognition-using-Open-CV-\trainingImages\3'}]

def train(absolutePathModel,list):
   test = os.path.exists(path_of_model)
   if(not test):
       faces,faceID=fr.labels_for_training_data('trainingImages')
       face_recognizer=fr.train_classifier(faces,faceID)
       face_recognizer.write('trainingData.yml')
#    else:
#         face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#         face_recognizer.read('trainingData.yml')


def identify( imagePath , modelPath):
    dict={}
    recognizerModel=cv2.face.LBPHFaceRecognizer_create()
    recognizerModel.read(modelPath)
    test_img = cv2.imread(imagePath)
    faces_detected,gray_img=fr.faceDetection(test_img)
    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+h]
        label,confidence=recognizerModel.predict(roi_gray)#predicting the label of given image
        dict['Confidence']=confidence
        dict['Label']=l1[label][label]
        return dict

        

train(path_of_model,l1)

Results = identify('TestImages/Darpan.jpg',path_of_model)

print(Results)



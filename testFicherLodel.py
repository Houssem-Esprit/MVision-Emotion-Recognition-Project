import cv2 
import numpy as np
import dlib

#emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
PredictDic = {0:"neutral",1:"anger",2:"contempt",3:"disgust",4:"fear",5:"happy",6:"sadness",7:"surprise"}

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
detector = dlib.get_frontal_face_detector()

def DetectandResizeImg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Detect face using 4 different classifiers
    """
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = ""
    """

    facefeatures=detector(gray, 1)
    #Cut and save face
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
        print ("face found in file")
        gray = gray[y:y+h, x:x+w] #Cut the frame to size
        filename = 'savedImage.jpg'   
        try:
            out = cv2.resize(gray, (48, 48)) #Resize face so all images have same size
            #cv2.imwrite("savedImage.jpg", outt) #Write image
            #out = cv2.imread("savedImage.jpg")
            print("out in the function: ", out)
            return out
        except:
            pass #If error, pass file
    

def test(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left() # left point
        y1 = face.top() # top point
        x2 = face.right() # right point
        y2 = face.bottom() # bottom point
        gray = gray[y1:y1+y2, x1:x1+x2]
        try:
            out = cv2.resize(gray, (48, 48)) #Resize face so all images have same size
            #cv2.imwrite("savedImage.jpg", outt) #Write image
            #out = cv2.imread("savedImage.jpg")
            print("out in the function: ", out)
            return out
        except:
            pass #If error, pass file


"""
print("List of Labels: ",PredictDic[0])

fishface = cv2.face.FisherFaceRecognizer_create()
fishface.read('trainedFicherModel.xml') 


img = cv2.imread("women.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

out = test(img)
cv2.imwrite("testt.jpg",out)
#print(out)

if(out is not None):
    prediction = fishface.predict(out)
    for i in PredictDic:
        if(i == prediction[0]):
            print("Predit with name",PredictDic[i])
    print("predicted emotion", prediction[0])
else:
    print('bad image to work with')
    pass

"""

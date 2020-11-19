import flask
from testFicherLodel import DetectandResizeImg,test
import cv2 
import numpy as np
import werkzeug

PredictDic = {0:"neutral",1:"anger",2:"contempt",3:"disgust",4:"fear",5:"happy",6:"sadness",7:"surprise"}



app = flask.Flask(__name__)



@app.route('/', methods=['GET', 'POST'])

def handle_request():

    imageFile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imageFile.filename)

    print("\nReceived image File name : " + imageFile.filename)
    print("ImageFile",imageFile)
    print("Filename",filename)

    imageFile.save(filename)
    img = cv2.imread("androidFlask.jpg")
    fishface = cv2.face.FisherFaceRecognizer_create()
    fishface.read('trainedFicherModel.xml') 
    #img = cv2.imread("womenHappy.jpg")
    out = test(img)
    print("This is the out",out)

    if(out is not None):
        prediction = fishface.predict(out)
        print('Prediction: RRRRRRR',prediction)
        for i in PredictDic:
            if(i == prediction[0]):
                print("Predit with name",PredictDic[i])
                return PredictDic[i]
        print("predicted emotion", prediction[0])
    else:
        print('bad image to work with')
        return 'bad image to work with'
        pass











app.run(host="localhost", port=3001, debug=True)
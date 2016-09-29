#!/usr/bin/env python2
import time
import json
start = time.time()
import cv2
import os
import pickle
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import copy
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

networkModel = os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7')

net = openface.TorchNeuralNet(
        networkModel,
        imgDim=96, # 96 is kept as img Dimension
        cuda=False)

dlibFacePredictor = os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat")

align = openface.AlignDlib(dlibFacePredictor)

classifierModel =  fileDir+'/../generated-embeddings/classifier.pkl'

email_id  = None
def send_mail(img_data, person):
    global email_id
    receiver = email_id
    try :
        msg = MIMEMultipart()
        msg['Subject'] = 'Face Detection'
        msg['From'] = 'mishra1995rashmi@gmail.com'
        msg['To'] = 'jindalshivam65@gmail.com'

        text = MIMEText(person)
        msg.attach(text)
        image = MIMEImage(img_data, name="blank")
        msg.attach(image)

        s = smtplib.SMTP('smtp.gmail.com',587) #port 465 or 587
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login('mishra1995rashmi@gmail.com','mishra1995')
        s.sendmail('mishra1995rashmi@gmail.com',receiver, msg.as_string())
        s.quit()
        print "mail sent for ", person
    except Exception as e :
        print e
        send_mail(img_data, person)

def getRep(bgrImg):
    global net, align
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    start = time.time()

    # Get the largest face bounding box
    # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None

    start = time.time()

    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                96, # args.imgDim is taken as 96
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")

    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))
    return reps


def infer(img):
    global classifierModel
    with open(classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)  # le - label and clf - classifer

    reps = getRep(img)
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print "No Face detected"
            return (None, None)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        # print predictions
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print str(le.inverse_transform(max2)) + ": "+str( predictions [max2])
        # ^ prints the second prediction
        confidences.append(predictions[maxI])

        # print("Predict {} with {:.2f} confidence.".format(person, confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences)


def findPersonInVideo(video_path, person=None, thresold = .5, url=None):
    global fileDir
    video_path = fileDir + '/../../../' + video_path
    print video_path
    file_name = video_path.split('/')[-1].split('.')[0]
    
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    if url :
        video_capture = cv2.VideoCapture(url)
    else :
        video_capture = cv2.VideoCapture(video_path)
    confidenceList = []
    frame_no = 1
    ret, frame = video_capture.read()
    while ret:
        # cv2.putText(frame, "P: {} C: {}".format(persons, confidences),
        #             (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1)
        print frame_no
        persons, confidences = infer(frame)

        #priting the output and logging
        # print "P: " + str(persons) + " C: " + str(confidences)
        if ((url and frame_no%5==0) or (not url)):
            if person:
                if person in persons:
                    index = persons.index(person)
                    if confidences[index] > thresold:
                        #logging output
                        output = {
                            'status' : 1,
                            'count' : frame_no,
                            'person' : person
                        }
                        with open(fileDir+'/../../'+'json/'+file_name+'.json', 'w') as file:
                            file.write(json.dumps(output))
                        cv2.imwrite('img.jpg', frame)
                        with open('img.jpg', 'rb') as file:
                            img_data = file.read()
                        os.remove('img.jpg')
                        print person, confidences[index]
                        send_mail(copy.deepcopy(img_data), person)

                    else :
                        output = {
                            'status' : 0,
                            'count': frame_no
                        }
                        with open(fileDir+'/../../'+'json/'+file_name+'.json', 'w') as file:
                            file.write(json.dumps(output))
                else :
                    output = {
                        'status' : 0,
                        'count': frame_no
                    }
                    with open(fileDir+'/../../'+'json/'+file_name+'.json', 'w') as file:
                        file.write(json.dumps(output))
            else :
                flag = True
                for i in xrange(len(persons)):
                    if confidences[i] > thresold:
                        flag  = False
                        output = {
                            'status' : 1,
                            'count' : frame_no,
                            'person' : persons[i]
                        }
                        with open(fileDir+'/../../'+'json/'+file_name+'.json', 'w') as file:
                            file.write(json.dumps(output))
                        cv2.imwrite('img.jpg', frame)
                        with open('img.jpg', 'rb') as file:
                            img_data = file.read()
                        print persons[i], confidences[i]
                        send_mail(copy.deepcopy(img_data), copy.deepcopy(persons[i]))
                if flag :
                    output = {
                        'status' : 0,
                        'count' : frame_no,
                    }
                    with open(fileDir+'/../../'+'json/'+file_name+'.json', 'w') as file:
                        file.write(json.dumps(output))
    
        if url:
            if frame_no%5==0:
                ret, frame = video_capture.read()
            frame_no += 1
        else :
            frame_no +=5
            video_capture.set(1,frame_no)
            ret, frame = video_capture.read()

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def predictPerson(image_path):
    global fileDir
    image_path = fileDir + '/../../../' + image_path
    images_name = image_path.split('/')[-1].split('.')[0]
    frame = cv2.imread(image_path)
    return infer(frame)


def main(email, video_path, image_path=None, url=None):
    global email_id
    email_id = email
    file_name = video_path.split('/')[-1].split('.')[0]
    if image_path:
        persons, confidences = predictPerson(image_path)
        print persons
        print confidences
        for i in xrange(len(persons)):
            if confidences[i] >.5:
                findPersonInVideo(video_path=video_path, person=persons[i], thresold=.5, url=url)
    else :
        findPersonInVideo(video_path=video_path, thresold = .65, url=url)
    output = {
        'status' : 2
    }
    with open(fileDir+'/../../'+'json/'+file_name+'.json', 'w') as file:
        file.write(json.dumps(output))    

# main('videos/download.jpg','videos/Srk aish.mp4')

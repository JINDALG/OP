#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
start = time.time()
import argparse
import cv2
import os
import pickle
from operator import itemgetter
import traceback
import numpy as np
np.set_printoptions(precision=2)
import pandas as pd
import smtplib
import openface
import threading
from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
face_found = False


def send_mail(receiver):
    msg = 'I found you guy... thank you'
    server = smtplib.SMTP('smtp.gmail.com',587) #port 465 or 587
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login('mishra1995rashmi@gmail.com','mishra1995')
    server.sendmail('mishra1995rashmi@gmail.com',receiver,msg)
    server.close()

def getRep(bgrImg):
    start = time.time()
    # bgrImg = cv2.imread(bgrImg)
    if bgrImg is None:
        pass
        # raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()
    reps=[]
    faces = align.getAllFaceBoundingBoxes(rgbImg)
    for bb in faces:
    # bb = align.getLargestFaceBoundingBox(rgbImg)
        if bb is None:
            # raise Exception("Unable to find a face: {}".format(imgPath)
            continue
            # return None
        # if args.verbose:
        #     print("Face detection took {} seconds.".format(time.time() - start))

        start = time.time()
        alignedFace = align.align(
            args.imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            print "face not found"
            # return "None"
            continue
            # raise Exception("Unable to align image: {}".format(imgPath))
        # if args.verbose:
        #     print("Alignment took {} seconds.".format(time.time() - start))

        start = time.time()
        rep = net.forward(alignedFace)
        # if args.verbose:
        #     print("Neural network forward pass took {} seconds.".format(
        #         time.time() - start))
        reps += [rep]
    return reps

def make_pridiction(id, img, clf, le):
        print 'start',id
        flag=False
        global face_found
        try :
            reps = getRep(img)
            for rep in reps:
                if rep == None:
                    # print "face not detected"
                    # return None
                    continue
                flag=True
                rep = rep.reshape(1,-1)
                start = time.time()
                predictions = clf.predict_proba(rep).ravel()
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = float(predictions[maxI])

                if person == "Akshay" and confidence >= .5:
                    print person," found with confidence == "  + str(confidence)
                    face_found = True
                    return
                print person," found with confidence == "  + str(confidence)
        except Exception as e:
            traceback.print_exc()
            # return None
            pass
        if not flag:
            print "face not found in frame"

class myThread (threading.Thread):
    def __init__(self, threadID, img, clf, le):
        threading.Thread.__init__(self)
        print "thread ",threadID," initalite"
        self.threadID = threadID
        self.img = img
        self.clf = clf
        self.le = le
    
    def run(self):
        make_pridiction(self.threadID, self.img, self.clf, self.le)

def infer(args):
    global face_found
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)
    # for img in args.imgs:
    #     make_pridiction(1, img, clf, le)

    for vid in args.imgs:
        cap = cv2.VideoCapture(vid)
        ############################################
        ############################################
        ## Video Loop
        ret, img = cap.read()
        threadID = 0
        frame_no = 1
        thread_list = []
        while(ret):
            ## Read the image
            ret, img = cap.read()
            cap.set(1,frame_no)
            frame_no += 5
            # if len(thread_list) == 15:
            #     if thread_list[0].isAlive():
            #         thread_list[0].join()
            #     thread_list.remove(thread_list[0])
            # thread = myThread(threadID, img, clf, le)
            # thread_list += [thread]
            # thread.start()
            threadID+=1
            make_pridiction(threadID, img, clf, le)
            if face_found == True:
                send_mail('nitinpawarme@gmail.com')                
                return
        # for i in xrange(len(thread_list)):
        #     if thread_list[i].isAlive():
        #         thread_list[i].join()
        print "person not found"
        # print("\n=== {} ===".format(img))
            
        # if args.verbose:
        #     print("Prediction took {} seconds.".format(time.time() - start))
        # print("Predict {} with {:.2f} confidence.".format(person, confidence))
        # if isinstance(clf, GMM):
        #     dist = np.linalg.norm(rep - clf.means_[maxI])
        #     print("  + Distance from the mean: {}".format(dist))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train',
                                        help="Train a new classifier.")
    trainParser.add_argument('--ldaDim', type=int, default=-1)
    trainParser.add_argument(
        '--classifier',
        type=str,
        choices=[
            'LinearSvm',
            'GMM',
            'RadialSvm',
            'DecisionTree',
            'GaussianNB',
            'DBN'],
        help='The type of classifier to use.',
        default='LinearSvm')
    trainParser.add_argument(
        'workDir',
        type=str,
        help="The input work directory containing 'reps.csv' and 'labels.csv'. Obtained from aligning a directory with 'align-dlib' and getting the representations with 'batch-represent'.")

    inferParser = subparsers.add_parser(
        'infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('imgs', type=str, nargs='+',
                             help="Input image.")

    args = parser.parse_args()
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(
            time.time() - start))

    if args.mode == 'infer' and args.classifierModel.endswith(".t7"):
        raise Exception("""
Torch network model passed as the classification model,
which should be a Python pickle (.pkl)

See the documentation for the distinction between the Torch
network and classification models:

        http://cmusatyalab.github.io/openface/demo-3-classifier/
        http://cmusatyalab.github.io/openface/training-new-models/

Use `--networkModel` to set a non-standard Torch network model.""")
    start = time.time()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                  cuda=args.cuda)

    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - start))
        start = time.time()

    
    infer(args)

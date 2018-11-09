###NOTE this file is for training documentation only
##This file is not relevant to the code operation

import numpy as np
import cv2
import os

#Training and test directories used
pathTrain = './Lets'
pathTest = './LetsR2'

#Lists initialisationa and alphabet lookup
alph = ['A', 'B', 'C', 'D', 'E', 'F','G','H','I','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
testImg = []
trainLbl = []
HOGDesc = []

winSize = (100, 120)
blockSize = (20,20)
blockStride = (10,10)
cellSize = (10,10)
nbins = 12
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True

#SVM Constants - Alphabet
gamma = 12
C = 50
#SVM Constants - Symbol
# gamma = 0.1625
# C = 100

#HOG Descriptor set-up
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)

#Load in file and append descriptor to list
for filename in os.listdir(pathTrain):
    imgIn = cv2.imread(pathTrain+'/'+filename, 0)
    #Determine what letter and append label to list
    trainLbl.append(alph.index(filename[0]))
    descriptor = hog.compute(imgIn)
    descriptor = descriptor.reshape(-1).astype(np.float32)
    HOGDesc.append(descriptor)

#Same as training set
for filename in os.listdir(pathTest):
    imgTest = cv2.imread(pathTest+'/'+filename, 0)
    h,w =  imgTest.shape
    descriptTest = hog.compute(imgTest)
    descriptTest = descriptTest.reshape(-1).astype(np.float32)
    testImg.append(descriptTest)

#Get lists in form for SVM predictor
HOGDesc = np.array(HOGDesc)
trainLbl = np.array(trainLbl)
testImg = np.array(testImg)

#Set model Parameters
model = cv2.ml.SVM_create()
model.setGamma(gamma)
model.setC(C)
model.setKernel(cv2.ml.SVM_LINEAR)
model.setType(cv2.ml.SVM_C_SVC)
model.train(HOGDesc, cv2.ml.ROW_SAMPLE, trainLbl)


#Save the model and predict against test data
model.save("textModel.xml")
results = model.predict(testImg)

#Output test data
i = 0
for i in range(len(results[1])):
    print(alph[int(results[1][i][0])])
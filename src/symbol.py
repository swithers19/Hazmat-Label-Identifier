import numpy as np
import cv2
import os

#Load symbol SVM
svmSymbol = cv2.ml.SVM_load('./Models/symbolModelR2.xml')

#HOG Parameters for Symbols
winSize = (320, 240)
blockSize = (20,20)
blockStride = (10,10)
cellSize = (10,10)
nbins = 8
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True

#Symbol Lookup table
SymbolLookup = ['1.4', '1.5','1.6','Corrosive', 'Explosive', 'Flame', 'Gas Cyclinder', 'Oxidiser', 'Radioactive','Skull and crossbones', 'Skull and crossbones on black diamond']
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)

class symbolRec:
    def __init__(self, croppedImage, filename):
        self.filename = filename
        w, h, c = croppedImage.shape
        #Make gray, crop and shape to allow HOG descriptor creation
        grayRoi = cv2.cvtColor(croppedImage,cv2.COLOR_BGR2GRAY)
        self.cropped = grayRoi[int(20):int(h/2), int(90):int(w-90)].copy()
        self.cropped = cv2.resize(self.cropped,(320,240),fx=0, fy=0, interpolation = cv2.INTER_CUBIC)

    def getSymbol(self):
        testImg = []
        symbol = getPrediction(hog, self.cropped, svmSymbol, SymbolLookup)
        return symbol


#Predictor function for the various HOG-SVG set-ups
def getPrediction(hogDescript, image, svmModel,lookupTable=None):
    testImg = []
    #Comput HOG descriptor
    descriptTest = hogDescript.compute(image)
    descriptTest = descriptTest.reshape(-1).astype(np.float32)
    testImg.append(descriptTest)
    descriptTest = np.array(testImg)
    #Predict with model
    prediction = svmModel.predict(descriptTest)
    #Look up correlating symbol or return relevant integer
    if lookupTable is not None:
        classification = lookupTable[int(prediction[1][0][0])]
    else:
        classification = str(int(prediction[1][0][0]))
    return classification

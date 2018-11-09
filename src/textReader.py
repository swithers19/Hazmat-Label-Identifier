import cv2
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import difflib
import symbol as sym

#Alphabet Lookup and SVM Load in
svmText = cv2.ml.SVM_load('./Models/textModel.xml')
alph = ['A', 'B', 'C', 'D', 'E', 'F','G','H','I','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

#HOG parameters
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

hogText = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)

class textReader:
    def __init__(self, croppedImage, filename, dictionary):
        self.filename = filename
        self.cropped = croppedImage 
        self.dictionary = dictionary

    #Determines if a valid word is present in string
    def identifyString(self, currentText):
        threshold =  0.6
        potentialWords = difflib.get_close_matches(currentText, self.dictionary, 3, threshold)
        if len(potentialWords)>0:
            while len(potentialWords)!=1:
                #increment threshold to limit list to one option
                if len(potentialWords)>1:
                    threshold+=0.05
                #Decrement threshold if no options remaining
                elif len(potentialWords)<1:
                    threshold-=0.005
                potentialWords = difflib.get_close_matches(currentText, self.dictionary, 3, threshold)
                #In case still many good options, get first in list
                if len(potentialWords)>1 and threshold>=0.95:
                    return potentialWords[1]
            return potentialWords[0]
        else:
            return "None"

    def findEdges(self):
        #Get ROI and increase size to improve ability to detect letters
        w, h, c = self.cropped.shape
        roi = self.cropped[int(h/2-60):int(h/2+70), int(30):int(w-30)].copy()
        roi = cv2.resize(roi, (0, 0), fx = 2.0, fy = 2.0, interpolation=cv2.INTER_LINEAR)

        #Grayscale, filter and normalise
        grayscaleROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bilatFilter = cv2.bilateralFilter(grayscaleROI, 7, 60, 10)
        im = cv2.normalize(bilatFilter, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        #Adaptive thresholding to identify text
        testingWhite = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,3)
        testingBlack = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21,3)
        
        #Combine images to extract text more clearly
        blackText = testingBlack & cv2.bitwise_not(testingWhite)
        whiteText = cv2.bitwise_not(testingBlack) & testingWhite
        blackCnt = np.count_nonzero(blackText)
        whiteCnt = np.count_nonzero(whiteText) 

        #this is used to find only text regions, remaining are ignored
        if (whiteCnt) < 30000:
            edgeAdapt = cv2.Canny(whiteText, 600, 650, apertureSize = 5, L2gradient = True)
        else:
            edgeAdapt = cv2.Canny(blackText, 600, 650, apertureSize = 5, L2gradient = True)
        return edgeAdapt


    #Function that returns text string for label
    def labelText(self):
        boundingBoxList = []
        thrBythrKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        
        #Edges and Text Contours found
        edges = self.findEdges()
        textContour = getContours(edges)
      
        #Get bounding boxes of contours for segmentation
        __, contourBoxes,__ = cv2.findContours(textContour.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        textContour = cv2.dilate(textContour, thrBythrKern)
        
        cnter = 0
        for contBox in contourBoxes:
            #cv2.drawContours(roi, [contBox], -1, (255, 0, 0), 3)
            xB, yB, wB, hB = cv2.boundingRect(contBox)
            boundingBoxList.append((xB, yB, wB, hB))
            cnter+=1

        #Words seperated by y-axis, identify text and apply any corrections    
        l1, l2 = getYSepWords(boundingBoxList)
        textString = getWords(l1, l2, textContour)
        return self.identifyString(textString)


#Get text contours from edge image
def getContours(edges):
    wRoi,hRoi = edges.shape
    textContour = np.zeros((wRoi, hRoi), np.uint8)
    contourList = []
    imgContour, contours, hierachy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)        
    #Get the contours we care about and add to list
    for contour in contours:
        arcLen = cv2.arcLength(contour, True)
        if arcLen>20:
            x,y,w,h = cv2.boundingRect(contour)
            boundingArea = w*h
            contourArea = cv2.contourArea(contour, False)
            aspect_ratio = float(h)/w
            boundingPerimeter = 2*w +2*h
            arcBoundRatio = arcLen/boundingPerimeter 
            #Below is a good contour
            if aspect_ratio>0.6 and contourArea/(boundingArea)>0.3 and contourArea>30:
                contourList.append((contourArea, contour))
    
    #Get max and remove noise before drawing on textContour
    try:
        maxContArea = max(contourList,key=lambda item:item[0])
    except:
        maxContArea = 10*wRoi*hRoi

    for cont in contourList:
        if cont[0]>(0.05*maxContArea[0]):
            cv2.drawContours(textContour, [cont[1]], -1, (255), 1)
    return textContour

#Seperates words based on a y-axis histogram
def getYSepWords(boundingList):
    yElements=[]
    boundingList.sort(key=itemgetter(1))
    for items in boundingList:
        yElements.append(items[1])

    n, bins, patches = plt.hist(yElements,8, alpha=0.5)
    peaks = np.where(n>2)[0]
    l1 = [] 
    l2 = []
    if (len(peaks)) >= 2:
        for j in range(len(peaks)-1):
            if (peaks[j+1]-peaks[j])>=2:
                l1, l2 = splitList(yElements, bins[j+1], boundingList)
            else:
                l1 = boundingList
    else:
        l1 = boundingList
    return(l1, l2)


#Splits a list based on y-axis split
def splitList(values, splitVal, boxList):
    listA = []
    listB = []
    i = 0
    for val in values:
        if val>splitVal:
            listA.append(boxList[i])
        else:
            listB.append(boxList[i])
        i+=1
    return listA,listB


#Gets words from a list of bounding boxes
def getWords(wordA, wordB, contourImage):
    labelText = ""
    wordA.sort(key=itemgetter(0))
    wordB.sort(key=itemgetter(0))
    i = 0
    if len(wordB)>0:
        #For multi-line words
        container = [wordB, wordA]
        for word in container:
            j = 0
            for box in word:
                xL,yL,wL,hL = box
                letter = contourImage[yL:yL+hL, xL:xL+wL].copy()
                letter = cv2.resize(letter,(100,120), 0, 0)
                labelText += sym.getPrediction(hogText, letter, svmText, alph)
    else:
        #For words on one line
        for box in wordA:
            xL,yL,wL,hL = box
            letter = contourImage[yL:yL+hL, xL:xL+wL].copy()
            letter = cv2.resize(letter,(100,120), 0, 0)
            labelText +=  sym.getPrediction(hogText, letter,svmText, alph)
        labelText += " "
    return labelText

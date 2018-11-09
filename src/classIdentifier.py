import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import symbol as sym
import textReader as tr

#Load symbol SVM
svmNumbers = cv2.ml.SVM_load('./Models/numbersModel.xml')

class ClassLabelID:
    def __init__(self, croppedImage, filename):
        self.filename = filename
        self.cropped = croppedImage 
    
    #Blob parameters
    def setParams(self):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 300
        params.maxArea = 1500
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        params.maxCircularity = 0.8
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.2
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.15
        return params

    def getKeypoints(self):
        #Get roi and two grayscale images
        w, h, c = self.cropped.shape
        roi = self.cropped[(h-150):h-10, int(w/2-60):int(w/2+60)].copy()
        grayscaleROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        invGrayscaleROI = cv2.bitwise_not(grayscaleROI)

        w1, h1 = grayscaleROI.shape

        #Get blobs based on parameters set
        blob = grayscaleROI.copy()
        blobB = invGrayscaleROI.copy()
        detector = cv2.SimpleBlobDetector_create(self.setParams())
        #Get blobs for both grayscale images
        keypointsA = detector.detect(blob)
        keypointsB =  detector.detect(blobB)

        return keypointsA, keypointsB, grayscaleROI, invGrayscaleROI

    #Isolates character and creates letter for classifier
    def isolateCharacter(self, keypoints, grayscaleROI):
        characterSet  = []
        if keypoints == None:
            return characterSet
        if len(keypoints)>0:
            for keyPoint in keypoints:
                x = keyPoint.pt[0]
                y = keyPoint.pt[1]
                s = keyPoint.size
                c = 0.3*s
                char = grayscaleROI[int(y-0.5*s-c):int(y+0.5*s+c),int(x-0.5*s):int(x+0.5*s)]
                char = cv2.resize(char,(100,120), 0, 0)
                thresh = cv2.threshold(char,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                characterSet.append(thresh)
        return characterSet

    #Builds class text and can handle >1 numbers
    def classLabelClassifier(self, characterSet):
        classNumbers = []
        classlbl = None
        for i in characterSet:
            holding = sym.getPrediction(tr.hogText,i,svmNumbers)
            if holding is not None:
                classNumbers.append(holding)

        #Dealing with different class number cases
        if len(classNumbers) == 0:
            classlbl = None
        elif len(classNumbers) == 2:
            if '5' in classNumbers:
                if '1' in classNumbers:
                    classlbl = '5.1'
                elif '2' in classNumbers:
                    classlbl = '5.2'
                else :
                    classlbl = None
        elif len(classNumbers) == 1:
            classlbl = str(classNumbers[0])
        return classlbl

    #Determines class label of
    def getClassLabel(self):
        ret  = None
        grayscaleImage = None
        #Get ROI
        w, h, c = self.cropped.shape
        roi = self.cropped[(h-150):h-10, int(w/2-60):int(w/2+60)].copy()
        grayscaleROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
       
        # Detect blobs.
        keypoints = self.getKeypoints()

        #im_with_keypoints = cv2.drawKeypoints(roi, keypoints[1], np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        blackBlob = keypoints[0]
        whiteBlob = keypoints[1]

        #Determine if black or white blob and pass to character isolator
        if (len(blackBlob) == 0 and len(whiteBlob) != 0):
            ret = whiteBlob
            grayscaleImage = keypoints[3]     
        elif (len(blackBlob) != 0 and len(whiteBlob) == 0):
            ret = blackBlob
            grayscaleImage = keypoints[2]     
        elif (len(blackBlob) != 0 and len(whiteBlob) != 0): 
            ret = blackBlob  
            grayscaleImage = keypoints[2]

        characterSet = self.isolateCharacter(ret, grayscaleImage)
        return self.classLabelClassifier(characterSet)


    
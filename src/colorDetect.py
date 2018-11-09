import cv2
import numpy as np
import os
import json

#defining colors
colorsObject = {
    'blue':{
        'lower': np.array([100,60,80]),
        'upper': np.array([130,180,255])
    },
    'orange': {
        'lower': np.array([5,140,100]),
        'upper': np.array([15,180,255])
    },
    'yellow':{
        'lower': np.array([15,90,40]),
        'upper':np.array([24,135,255])
    },
    'green':{
        'lower': np.array([24, 60, 90]),
        'upper': np.array([88, 140,255])
    },
    'red':{
        'lower': np.array([170, 60, 80]),
        'upper': np.array([180, 140,255])
    },
    'orange2':{
        'lower': np.array([175, 141, 80]),
        'upper': np.array([180, 190,255])
    },
    'red2':{
        'lower': np.array([0, 50, 80]),
        'upper': np.array([8, 135,255])
    }
}

blackWhiteObject ={
    'black':{
        'lower': np.array([0,0,0]),
        'upper':np.array([180,80,255])
    },
    'white':{
        'lower': np.array([0,155,0]),
        'upper':np.array([180,255,255])
    }
}


class colorDetect:
    def __init__(self, filename, croppedImage):
        self.filename = filename 
        self.image = croppedImage
        self.image = cv2.blur(self.image, (5, 5)) 
        self.hsl = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)
        h, w, c = self.hsl.shape      
        self.topCropped = self.hsl[0:int(h/2), 0:w]
        self.bottomCropped = self.hsl[int(h/2):h, 0:w]
        self.top = ''
        self.bottom = ''

    def determineColor(self):
        self.checkColors()
        self.blackWhite()
        if not (self.top is None):
            self.top = self.top.replace("2","")
        if not (self.bottom is None):
            self.bottom = self.bottom.replace("2","")
    
    #Determine if black or white after color
    def blackWhite(self):
        bw = []
        i=0
        currentVal = (self.top, self.bottom)
        for section in currentVal:
            if section ==None:
                if i == 0:
                    self.top = self.checkColorInHalf(blackWhiteObject, 1)
                if i ==1:
                    self.bottom = self.checkColorInHalf(blackWhiteObject, 0)
            i+=1

    def checkColorInHalf(self, colObj, topFlag):
        if topFlag:
            roiCrop = self.topCropped
        else:
            roiCrop = self.bottomCropped

        for color,colRange in colObj.items():
            if color == 'black':
                thresh = 0.7
            elif color =='white':
                thresh = 0.12
            else:
                thresh = 0.1
            mask = cv2.inRange(roiCrop,colRange['lower'], colRange['upper'])
            onesCnt = cv2.countNonZero(mask)
            if onesCnt/mask.size > thresh:
                return color
            
    def checkColors(self):
        for i in range(2):
            if i == 0:
                self.top = self.checkColorInHalf(colorsObject, 1)
            if i ==1:
                self.bottom = self.checkColorInHalf(colorsObject, 0)
            i+=1

    def getColor(self):
        imgColor = {
            'file':self.filename,
            'top':self.top,
            'bottom':self.bottom
        }
        return imgColor

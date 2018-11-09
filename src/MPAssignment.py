import cv2
import numpy as np
import pytesseract
import os

import getLabel as lf
import colorDetect as cd
import classIdentifier as tr
import textReader as textRead
import symbol as symRec

#define path to photos
path = './'
labelText = []

#Dictionary
with open('./Models/dictionary.txt') as fp:
    labelTexts = fp.readlines()    
for text in labelTexts:
    trimmedText = text.rstrip()
    labelText.append(trimmedText)

#Get files in a directory
for filename in os.listdir(path):
    #Get only image files
    if filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".jpg"):
        #Image segmentation
        labelShape = lf.LabelShapeIdentifier(filename, path)
        cropImg = labelShape.GetROI()
        if cropImg is None:
            print("Label not found")
        else:
            #Class label classification
            clssLblID = tr.ClassLabelID(cropImg, filename)
            classLabel = clssLblID.getClassLabel()
            #Colour classification
            imgColour = cd.colorDetect(filename, cropImg.copy())
            imgColour.determineColor()
            colour = imgColour.getColor()
            #Text Reader
            textReader = textRead.textReader(cropImg, filename, labelText)
            text = textReader.labelText()
            #Symbol Recognition
            symbolRecog = symRec.symbolRec(cropImg, filename)
            symbol = symbolRecog.getSymbol()

            #printing output
            print(filename)
            print("top: " + str(colour['top']))
            print("bottom: " + str(colour['bottom']))
            print("class: " + str(classLabel))
            print("text: "  + str(text))
            print("symbol: " + str(symbol))
            print()




    








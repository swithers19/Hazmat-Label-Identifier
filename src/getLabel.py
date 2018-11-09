import numpy as np
import cv2
import os


class LabelShapeIdentifier:
    def __init__(self, filename, path):
        fileDir = os.path.join(path, filename)
        self.filename = filename
        self.image = cv2.imread(fileDir)
        blurred = cv2.GaussianBlur(self.image, (5,5), 0)
        self.imG = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
        self.edges = self.getEdges()
        self.ROIList = self.CalculateROIs()
        self.roi = self.drawRegions()
        

    def getEdges(self):
        gray = cv2.bitwise_not(self.imG)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115,1)

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        postNoise = cv2.bitwise_and(thresh, gray)
       
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE()
        contCorr = clahe.apply(postNoise)
        
        #Get prominent edges and dilate
        edgey = cv2.Canny(contCorr, 800, 850, apertureSize = 5, L2gradient = True) 
        edgey= cv2.dilate(edgey, kernel, iterations = 1)
        edgey = cv2.medianBlur(edgey, 5)

        return edgey
    
    #Gets ROI's of labels 
    def CalculateROIs(self):
        h, w = self.imG.shape
        markers, contours, hierachy = cv2.findContours(self.edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        myList = []
        locationA = (0,0)
        for contour in contours:
            #If within reasonable bounds of image
            if (cv2.contourArea(contour)> 0.1*h*w) & (cv2.contourArea(contour)<0.9*h*w):
                perimeter = cv2.arcLength(contour, True)
                poly = cv2.approxPolyDP(contour, 0.1*perimeter, True)  
                #Ensure contour has 4 vertices
                if len(poly) == 4:
                    areaPoly = cv2.contourArea(poly)    
                    location, dims, theta = cv2.minAreaRect(contour)
                    boundingRectArea = dims[0]*dims[1]
                    #Ensure the bounding rect and poly and roughly similar
                    if boundingRectArea/areaPoly<1.25:
                        difX = float(location[0]-locationA[0])/w
                        difY = float(location[1]-locationA[1])/h
                        #Remove internal contours and add polyDP to list
                        if ((difX>0.1) | (difX<=-0.1)) | ((difY>0.1) | (difY<=-0.1)):  
                            locationA = location                                      
                            myList.append([poly, dims, location]) 
        return myList

    def getROIs(self):
        return self.ROIList
    
    def drawRegions(self):
        shape = self.image.shape
        lst = []
        #Designed with goal for multiple labels
        for x in self.ROIList:
            #Find the corresponding ROI based on the poitns
            y = cv2.convexHull(x[0])                        
            holding = np.zeros((shape[0], shape[1]), np.uint8)
            holding = cv2.fillConvexPoly(holding, y, (255))   
            roi = cv2.bitwise_and(self.image, self.image, mask=holding)
            maxVals = np.max(y, axis=0)
            minVals = np.min(y, axis=0)
            #Affine transform label to diamond
            pts1 = np.float32(y[:3])
            pts2 = np.array([[400,200], [200, 400], [0, 200]],  np.float32)
            M = cv2.getAffineTransform(pts1,pts2)
            warp = cv2.warpAffine(roi,M,(400,400))
            return warp

    def GetROI(self):
        return self.roi







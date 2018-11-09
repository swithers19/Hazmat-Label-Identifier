import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

pathTrain = './TrainPrep'
pathTest = './Test'

testImg = []
trainLbl = []
HOGDesc = []

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

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)


i = 0
for filename in os.listdir(pathTrain):
    print(filename)
    imgIn = cv2.imread(pathTrain+'/'+filename, 0)
    #, h = imgIn.shape
    img = imgIn.copy()
    print(imgIn.shape)
    descriptor = hog.compute(imgIn)
    #print(descriptor)
    descriptor = descriptor.reshape(-1).astype(np.float32)
    HOGDesc.append(descriptor)

for filename in os.listdir(pathTest):
    imgTest = cv2.imread(pathTest+'/'+filename, 0)
    h,w =  imgTest.shape
    imgTest = imgTest[int(20):int(h/2), int(90):int(w-90)].copy()
    imgTest = cv2.resize(imgTest,(320,240),fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
    print(imgTest.shape)
    descriptTest = hog.compute(imgTest)
    descriptTest = descriptTest.reshape(-1).astype(np.float32)
    print(descriptTest)
    testImg.append(descriptTest)


k = np.arange(10)
train_labels = np.repeat(k,4)[:,np.newaxis]

print(np.repeat(1,4)[:,np.newaxis])
trainLbls = np.array([0, 0, 0, 1, 1, 1, 1, 1,2, 2, 2, 2, 2, 2,3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
4, 4, 4, 4, 4, 4, 
5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5, 5, 5, 5, 5, 5,5,
6, 6, 6, 6, 6, 6,
7, 7, 7, 7, 7, 7,
8, 8, 8, 8, 8, 8, 8, 8,
9, 9, 9, 9, 9, 9, 9, 9,
10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
print(len(trainLbls))
#trainLbls = np.array([0, 0, 0, 1, 1, 1, 1,1, 2, 2, 2,2,2,2,3, 3, 3, 3, 3, 3,3,3,4, 4, 4, 4,4,5,5, 5, 5, 5, 5, 5, 5,5, 5, 5, 5, 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6, 6, 6, 6,6,6,7, 7, 7, 7,7,7,8, 8, 8,8,8,8,8,8,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10])
print(trainLbls)
HOGDesc = np.array(HOGDesc)

testImg = np.array(testImg)
print(np.shape(HOGDesc))
print(np.shape(testImg))

gamma = 0.1625
C = 100
model = cv2.ml.SVM_create()
model.setGamma(gamma)
model.setC(C)
model.setKernel(cv2.ml.SVM_LINEAR)
model.setType(cv2.ml.SVM_C_SVC)
model.train(HOGDesc, cv2.ml.ROW_SAMPLE, trainLbls)
#model.trainAuto(HOGDesc, cv2.ml.ROW_SAMPLE,train_labels, kFold=3)

model.save("symbolModelR2.xml")
results = model.predict(testImg)
print(results[1])
# for filename in os.listdir(pathTest):
#     img = cv2.imread(pathTest+'/'+filename, 0)
#     interm = img.reshape(-1).astype(np.float32)
#     testImg.append(interm)

# # trainImg = np.asarray(trainImg)
# # testImg = np.asarray(testImg)
# # trainLbl = np.asarray(trainLbl)
# # test_labels = trainLbl
# # print(np.shape(trainImg))



# knn = cv2.ml.KNearest_create()
# knn.train(trainImg, cv2.ml.ROW_SAMPLE, trainLbl)
# ret,result,neighbours,dist = knn.findNearest(testImg,k=5)

# # Now we check the accuracy of classification
# # For that, compare the result with test_labels and check which are wrong
# matches = result==test_labels
# correct = np.count_nonzero(matches)
# accuracy = correct*100.0/result.size
# print(accuracy)
# #     svm = cv2.SVM()
# #     svm.train(trainData,responses, params=svm_params)
# # svm.save('svm_data.dat')
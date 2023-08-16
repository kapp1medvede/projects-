import glob
import numpy as np
import cv2 as cv
import joblib
import os
from skimage.feature import hog

class SodukoSolver :

    def init(self , img):

        self.img = img

        # convert the image to gray scale image
        self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.gray = cv.resize(self.gray,(567,567))

    def imageProcessing(self):

        # Gaussian Blur to decrease noise
        gaussianBlur = cv.GaussianBlur(self.gray, (5, 5), 0)

        # Threshold to get binary image
        adaptiveThreshold = cv.adaptiveThreshold(gaussianBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11,
                                                 2)

        # Median Blur to decrease salt and peper noise
        medianBlur = cv.medianBlur(adaptiveThreshold, 3)
        medianBlur = cv.medianBlur(medianBlur, 3)

        # opening and closing to get better result
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(medianBlur, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

        # inverting
        invert = (255 - closing)

        return invert

    def gridImageExtraction(self , binaryImage):

        rows, cols = self.gray.shape[:2]
        dst_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])

        # find all contours over the image after converting it to binary form
        _ , self.contours , self.hierarchy  = cv.findContours(binaryImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # generate a list of tuples gather the contour and its index
        # index is useful to get the children of specific contour by use the (self.hierarchy)
        # find in tutorial
        self.sortedContours = [(contour,i) for i,contour in enumerate(self.contours)]
        # sort this list by contour area
        self.sortedContours = sorted(self.sortedContours,
                                     key= lambda contour :  cv.contourArea(contour[0]), reverse=True)

        # iterate over the first 5 contours
        for i in range(5):
            # get approximated contour
            peri = cv.arcLength(self.sortedContours[i][0], True)
            approx = cv.approxPolyDP(self.sortedContours[i][0], 0.018 * peri, True)

            if len(approx) == 4 :
                projectionPointsList = self.getProjectionPointsList(approx)

                # make projection to the contour over all the image
                src_points = np.float32(projectionPointsList)
                projective_matrix = cv.getPerspectiveTransform(src_points, dst_points)
                grid = cv.warpPerspective(binaryImage, projective_matrix, (cols, rows))
                if (self.gridVerification(grid,self.sortedContours[i][1])):
                    return cv.warpPerspective(self.gray, projective_matrix, (cols, rows))
        return None

    def getProjectionPointsList(self , approximatedContour):
        approximatedContour = approximatedContour.ravel()
        approximatedContour = approximatedContour.reshape((4, 2))
        ind = np.argsort(approximatedContour[:, -1])
        points = approximatedContour[ind]

        projectionPointsList = [[], [], [], []]
        cnt = 3
        if (points[0][0] < points[1][0]):
            points[0][0] -= cnt ; points[1][0] += cnt
            points[0][1] -= cnt ; points[1][1] -= cnt
            projectionPointsList[0].append(points[0])
            projectionPointsList[1].append(points[1])
        else:
            points[1][0] -= cnt; points[0][0] += cnt
            points[1][1] -= cnt; points[0][1] -= cnt
            projectionPointsList[0].append(points[1])
            projectionPointsList[1].append(points[0])

        if (points[2][0] < points[3][0]):
            points[2][0] -= cnt; points[3][0] += cnt
            points[2][1] += cnt; points[3][1] += cnt
            projectionPointsList[2].append(points[2])
            projectionPointsList[3].append(points[3])
        else:
            points[3][0] -= cnt; points[2][0] += cnt
            points[3][1] += cnt; points[2][1] += cnt
            projectionPointsList[2].append(points[3])
            projectionPointsList[3].append(points[2])

        return projectionPointsList

    def gridVerification(self,grid,level):

        # hasn't completed yet
        cv.imshow("Grid", grid)
        cv.waitKey(0)

        areas = []
        for i , index in enumerate(self.hierarchy[0]):
            if index[3]  == level :
                peri = cv.contourArea(self.contours[i])
                areas.append(peri)
                cv.drawContours(self.gray , [self.contours[i]],0 , (0,0,0),5)
                cv.imshow('sdddd',self.gray)
                cv.waitKey()

        areas = sorted(areas,reverse=True)
        return False

    def solve(self , image):
        self.init(image)
        invertedBinaryImage = self.imageProcessing()
        grid = self.gridImageExtraction(invertedBinaryImage)

        if (grid is None):
            print ('Grid is not existed')
        else :
            cv.imshow("Grid",grid)
            cv.waitKey()

img = cv.imread('so.jpg')
soduckSolver = SodukoSolver()
soduckSolver.solve(img)
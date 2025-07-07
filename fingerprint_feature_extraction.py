import cv2
import numpy as np
import skimage.morphology
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import footprint_rectangle
from skimage import draw, measure
import math 

class MinutiaeFeature(object):
    def __init__(self, locX, locY, Orientation, Type):  # Corrected constructor
        self.locX = locX
        self.locY = locY
        self.Orientation = Orientation
        self.Type = Type

class FingerprintFeatureExtractor(object):
    def __init__(self):  # Corrected constructor
        self._mask = []
        self._skel = []
        self.minutiaeTerm = []
        self.minutiaeBif = []
        self._spuriousMinutiaeThresh = 10

    def setSpuriousMinutiaeThresh(self, spuriousMinutiaeThresh):
        self._spuriousMinutiaeThresh = spuriousMinutiaeThresh

    def __skeletonize(self, img):
        img = np.uint8(img > 128)
        self._skel = skimage.morphology.skeletonize(img)
        self._skel = np.uint8(self._skel) * 255
        self._mask = img * 255

    def __computeAngle(self, block, minutiaeType):
        angle = []
        (blkRows, blkCols) = np.shape(block)
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        if minutiaeType.lower() == 'termination':
            for i in range(blkRows):
                for j in range(blkCols):
                    if (i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0:
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
            if len(angle) > 1:
                return [float('nan')]
        elif minutiaeType.lower() == 'bifurcation':
            for i in range(blkRows):
                for j in range(blkCols):
                    if (i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0:
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
            if len(angle) != 3:
                return [float('nan')]
        return angle

    def __getTerminationBifurcation(self):
        self._skel = self._skel == 255
        (rows, cols) = self._skel.shape
        self.minutiaeTerm = np.zeros(self._skel.shape)
        self.minutiaeBif = np.zeros(self._skel.shape)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if self._skel[i][j] == 1:
                    block = self._skel[i - 1:i + 2, j - 1:j + 2]
                    block_val = np.sum(block)
                    if block_val == 2:
                        self.minutiaeTerm[i, j] = 1
                    elif block_val == 4:
                        self.minutiaeBif[i, j] = 1

        self._mask = convex_hull_image(self._mask > 0)
        self._mask = erosion(self._mask, footprint_rectangle((5, 5)))
        self.minutiaeTerm = np.uint8(self._mask) * self.minutiaeTerm

    def __removeSpuriousMinutiae(self, minutiaeList, img):
        img = img * 0
        SpuriousMin = []
        numPoints = len(minutiaeList)
        D = np.zeros((numPoints, numPoints))
        for i in range(1, numPoints):
            for j in range(0, i):
                (X1, Y1) = minutiaeList[i]['centroid']
                (X2, Y2) = minutiaeList[j]['centroid']
                dist = np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)
                D[i][j] = dist
                if dist < self._spuriousMinutiaeThresh:
                    SpuriousMin.append(i)
                    SpuriousMin.append(j)

        SpuriousMin = np.unique(SpuriousMin)
        for i in range(numPoints):
            if i not in SpuriousMin:
                (X, Y) = np.int16(minutiaeList[i]['centroid'])
                img[X, Y] = 1

        img = np.uint8(img)
        return img

    def __cleanMinutiae(self, img):
        self.minutiaeTerm = measure.label(self.minutiaeTerm, connectivity=2)
        RP = measure.regionprops(self.minutiaeTerm)
        self.minutiaeTerm = self.__removeSpuriousMinutiae(RP, np.uint8(img))

    def __performFeatureExtraction(self):
        FeaturesTerm = []
        self.minutiaeTerm = measure.label(self.minutiaeTerm, connectivity=2)
        RP = measure.regionprops(np.uint8(self.minutiaeTerm))

        WindowSize = 2
        for i in RP:
            (row, col) = np.int16(np.round(i.centroid))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Termination')
            if len(angle) == 1:
                FeaturesTerm.append(MinutiaeFeature(row, col, angle[0], 'Termination'))

        FeaturesBif = []
        self.minutiaeBif = measure.label(self.minutiaeBif, connectivity=2)
        RP = measure.regionprops(np.uint8(self.minutiaeBif))
        WindowSize = 1
        for i in RP:
            (row, col) = np.int16(np.round(i.centroid))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Bifurcation')
            if len(angle) == 3:
                FeaturesBif.append(MinutiaeFeature(row, col, angle[0], 'Bifurcation'))
        return FeaturesTerm, FeaturesBif

    def extractMinutiaeFeatures(self, img):
        self.__skeletonize(img)
        self.__getTerminationBifurcation()
        self.__cleanMinutiae(img)
        return self.__performFeatureExtraction()

    def showResults(self, FeaturesTerm, FeaturesBif):
        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255 * self._skel
        DispImg[:, :, 1] = 255 * self._skel
        DispImg[:, :, 2] = 255 * self._skel

        for curr_minutiae in FeaturesTerm:
            row, col = curr_minutiae.locX, curr_minutiae.locY
            rr, cc = draw.circle_perimeter(row, col, 3)
            draw.set_color(DispImg, (rr, cc), (0, 0, 255))

        for curr_minutiae in FeaturesBif:
            row, col = curr_minutiae.locX, curr_minutiae.locY
            rr, cc = draw.circle_perimeter(row, col, 3)
            draw.set_color(DispImg, (rr, cc), (255, 0, 0))

        cv2.imshow("Fingerprint Features", DispImg)
        cv2.waitKey(0)

    def saveResult(self, FeaturesTerm, FeaturesBif):
        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255 * self._skel
        DispImg[:, :, 1] = 255 * self._skel
        DispImg[:, :, 2] = 255 * self._skel

        for idx, curr_minutiae in enumerate(FeaturesTerm):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

        for idx, curr_minutiae in enumerate(FeaturesBif):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))
        cv2.imwrite('result.png', DispImg)

def extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False):
    feature_extractor = FingerprintFeatureExtractor()
    feature_extractor.setSpuriousMinutiaeThresh(spuriousMinutiaeThresh)
    if (invertImage):
        img = 255 - img;

    FeaturesTerm, FeaturesBif = feature_extractor.extractMinutiaeFeatures(img)

    if (saveResult):
        feature_extractor.saveResult(FeaturesTerm, FeaturesBif)

    if(showResult):
        feature_extractor.showResults(FeaturesTerm, FeaturesBif)

    return(FeaturesTerm, FeaturesBif)
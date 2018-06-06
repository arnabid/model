# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:27:10 2017

@author: H128815 <yogendra.pandey@halliburton.com>
BRAINS4Reservoirs - Petrophysical Properties Regression.
"""
# Import the required modules.
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler

"""
This class defines Generalized Regression Neural Network (GRNN), 
which facilitates interpolation of the petrophysical properties 
along the depth of wells. This is an adaptation of GRNN definition
outlined in:
D. F. Specht, "A general regression neural network", IEEE trans. 
Neural Networks, vol. 2, pp. 568-576, Nov. 1991.
"""



class GRNN:
    """
    Constructor for a GRNN.
    """
    def __init__(self, xi, yi, sig):
        assert xi.shape[0] == yi.shape[0]
        self.X = xi
        self.Y = yi
        self.sigma = sig
        self.sigsq = 2.0 * sig**2

    """
    This function implements prediction algorithm for the GRNN.
    Parameters:
    ds - depths to calculate interpolated property values.
    """
    def predict(self, ds):
        res = []
        for d in ds:
            D2 = np.multiply((self.X - d), (self.X - d))
            numerator = np.sum(np.multiply(self.Y, np.exp(np.divide(-D2, self.sigsq))), axis=0)
            denominator = np.sum(np.exp(np.divide(-D2, self.sigsq)), axis=0)
            yp = numerator / denominator
            res.append(yp)
        return res


class ParallelInputPC:
    """
    Constructor for input used by feature creation from point cloud data.
    """
    def __init__(self, x, y, z, zShift, wellXY, wellZMin, wellZMax, grnn, 
                 wellPropMin, wellPropMax, nNeighborWells, sill,
                 vRange, hRange, vNugget, hNugget):
        self.x = x
        self.y = y
        self.z = z
        self.zShift = zShift
        self.wellXY = wellXY
        self.wellZMin = wellZMin
        self.wellZMax = wellZMax
        self.grnn = grnn
        self.wellPropMin = wellPropMin
        self.wellPropMax = wellPropMax
        self.nNeighborWells = nNeighborWells
        self.sill = sill
        self.vRange = vRange
        self.hRange = hRange
        self.vNugget = vNugget
        self.hNugget = hNugget

class B4RRegressor:
    """
    Constructor for B4RRegressor.
    Parameters:
    inputPath - location of the file containing data for building model
    """
    def __init__(self, inputPath, xColName, yColName, zColName, propColName, \
                 wellColName, seed, sill, hNugget, hRange, vNugget, vRange, \
                 shiftZ, nNeighborWells, xyzNDV, propNDV):
        self.inputPath = inputPath
        self.xColName = xColName
        self.yColName = yColName
        self.zColName = zColName
        self.propColName = propColName
        self.wellColName = wellColName
        self.seed = seed
        self.sill = sill
        self.hNugget = hNugget
        self.hRange = hRange
        self.vNugget = vNugget
        self.vRange = vRange
        self.nNeighborWells = nNeighborWells
        self.nSamples = -1
        self.uniqueWells = None
        self.nPointsWell = None
        self.wells = None
        self.x = None
        self.y = None
        self.z = None
        self.prop = None
        self.wellXY = []
        self.wellZ = []
        self.wellZMin = []
        self.wellZMax = []
        self.wellProp = []
        self.wellNeighbors = []
        self.wellPairsDist = None
        self.grnns = []
        self.zShift = self.vRange * shiftZ
        self.xyzNDV = xyzNDV
        self.propNDV = propNDV
        self.scaler = None
        self.propMean = None
        self.propMin = None
        self.propMax = None

    """
    This function provides computed semi-variance based
    upon the passed value of lag variable using a sperical
    variogram model equation.
    Parameters:
    nugget - nugget variance
    sill - variogram sill
    r - variogram range
    h - Lag variable
    """
    def sphericalModel(self, nugget, r, h):
        if (h >= r):
            return self.sill
        else:
            return ((self.sill - nugget) * 
                    ((1.5 * h / r) - 0.5 * (h / r)**3) + nugget)

    """
    This method formulates input feature matrix and output vector
    for training the deep neural network based upon the data in file 
    located at the provided inputPath.
    """
    def formulateInputOutput(self):
        # Importing the dataset as pandas dataframe
        df = pd.read_csv(self.inputPath)

        # Getting index of relevant columns (integers)
        wellsCol = df.columns.get_loc(self.wellColName)
        xCol = df.columns.get_loc(self.xColName)
        yCol = df.columns.get_loc(self.yColName)
        zCol = df.columns.get_loc(self.zColName)
        propCol = df.columns.get_loc(self.propColName)

        # Extracting relevant column vectors (numpy ndarrays)
        # well IDs
        self.wells = df.iloc[:, wellsCol].values.reshape(-1, 1).astype('float32')

        # x locations
        self.x = df.iloc[:, xCol].values.reshape(-1, 1).astype('float32')

        # y locations
        self.y = df.iloc[:, yCol].values.reshape(-1, 1).astype('float32')

        # z locations
        self.z = df.iloc[:, zCol].values.reshape(-1, 1).astype('float32')

        # property values
        self.prop = df.iloc[:, propCol].values.reshape(-1, 1).astype('float32')

        # Extract the valid data points in the current interval
        idxs = np.where(np.logical_and(self.x != self.xyzNDV, self.prop != self.propNDV))[0]
        self.wells = self.wells[idxs]
        self.x = self.x[idxs]
        self.y = self.y[idxs]
        self.z = self.z[idxs]
        self.prop = self.prop[idxs]

        # Initialize well information
        self.initializeWellsInfo(self.x, self.y, self.z, self.prop)

        self.nSamples = self.wells.shape[0]
        # Formulate input feature - Serial processing
        features = []
        for i in range(self.nSamples):
            feature = self.getInputFeatureForPoint(i, self.z[i][0])
            features.append(feature)
        features = np.array(features).astype('float32')
        self.prop = np.array(self.prop).reshape(-1,1).astype('float32')

        # Ensure that input feature doesn't contain any NaN or +/-Inf values
        self.propMean = np.mean(self.prop)
        self.propMin = np.min(self.prop)
        self.propMax = np.max(self.prop)
        idxs = np.where(np.logical_or(np.isnan(features), np.isinf(features)))
        features[idxs] = self.propMean

        # Define the min-max scaler
        self.scaler = MinMaxScaler()
        #self.scaler.fit(self.prop)
        self.scaler.fit(features)

        return features, self.prop

    """
    This function initializes information related to the wells,
    which is later on used for input feature formulation.
    Parameters:
    x - list containing x-coordinates
    y - list containing y-coordinates
    z - list containing z-coordinates
    prop - list containing property values
    """
    def initializeWellsInfo(self, x, y, z, prop):
        # self.uniqueWells - unique well IDs in sorted order
        # self.nPointsWell - counts of each well ID
        self.uniqueWells, self.nPointsWell = np.unique(self.wells, return_counts=True)

        # number of unique well IDs
        nWells = len(self.uniqueWells)

        for i in range(nWells):
            # indices of 'ith' unique wellID
            idxs = np.where(self.wells == self.uniqueWells[i])

            # first index where the 'ith' unique wellID is found
            idx = np.argmax(self.wells == self.uniqueWells[i])

            # self.wellXY - list of starting [[x,y],[],..] of each unique wellID
            self.wellXY.append([x[idx], y[idx]])

            # self.wellZ - list of z values for each unique wellID  
            self.wellZ.append(z[idxs])

            # self.wellZMin - list of min z value for each unique wellID
            self.wellZMin.append(np.min(self.wellZ[i]))

            # self.wellZMax - list of max z value for each unique wellID
            self.wellZMax.append(np.max(self.wellZ[i]))

            # self.wellProp - list of property values for each unique wellID 
            self.wellProp.append(prop[idxs])

        # calculate pairwise distance between all the unique wellIDs
        self.wellXY = np.array(self.wellXY).reshape(-1, 2)
        self.wellPairsDist = squareform(pdist(self.wellXY))

        # Populate nearest neighbor well excluding hold-out well
        for i in range(nWells):
            # self.wellNeighbors - indices of n nearest wells excluding the holdOut test well
            self.wellNeighbors.append(self.getNearestNeighbors(i))
        
        # initialize a GRNN for each unique well
        self.initGRNNs(nWells)
        
    """
    This function returns the 'n' nearest neighbor wells for the i-th
    well, excluding the hold-out test well.
    Parameters:
    iWell - well index for which neighbors are computed
    """
    def getNearestNeighbors(self, iWell):
        # Get neighbor indexes sorted by distances
        idxs = np.argsort(self.wellPairsDist[iWell])[1:]
        return idxs[0:self.nNeighborWells]

    """
    This function initializes a list of GRNNs for interpolating
    the property values along each well.
    Parameters:
    nWells - number of wells in the input data
    """
    def initGRNNs(self, nWells):
        for i in range(nWells):
            grnn = GRNN(self.wellZ[i], self.wellProp[i], np.std(self.wellProp[i]))
            self.grnns.append(grnn)

    """
    This function returns the interpolated property value,
    and corresponding effective z-value for calculation.
    If the value lies beyond the depth range in available
    well log, then the closest property value is returned.
    Parameters:
    z - depth along well to predict property
    i - index of the well for interpolation
    """
    def calcPropValue(self, z, i):
        if z > self.wellZMin[i] and z < self.wellZMax[i]:
            return self.grnns[i].predict(np.array([z]))[0], z
        elif z <= self.wellZMin[i]:
            return self.wellProp[i][0], self.wellZMin[i]
        elif z >= self.wellZMax[i]:
            return self.wellProp[i][-1], self.wellZMax[i]

    """
    This function returns the input feature for the point index
    i passed as the parameter.
    Parameters:
    i - index of the well containing the point
    z - depth of the point to create input feature
    """
    def getInputFeatureForPoint(self, i, z):
        feat = []

        # get the index of the wellID where this sample is located
        iWell = np.argmax(self.uniqueWells == self.wells[i])
        for jWell in self.wellNeighbors[iWell]:
            prop, zEff = self.calcPropValue(z, jWell)
            feat.append(prop)
            
            dxy = self.wellPairsDist[iWell][jWell]
            varH = self.sphericalModel(self.hNugget, self.hRange, dxy)
            dz = abs(z - zEff)
            varV = self.sphericalModel(self.vNugget, self.vRange, dz)
            feat.append(prop + np.sqrt(2 * varH))
            feat.append(prop + np.sqrt(2 * varV))
            
            propUp, zEffUp = self.calcPropValue(z + self.zShift, jWell)
            feat.append(propUp )
            dzUp = abs(z - zEffUp)
            varVUp = self.sphericalModel(self.vNugget, self.vRange, dzUp)
            feat.append(propUp + np.sqrt(2 * varH))
            feat.append(propUp + np.sqrt(2 * varVUp))
            
            propDn, zEffDn = self.calcPropValue(z - self.zShift, jWell)
            feat.append(propDn)
            dzDn = abs(z - zEffDn)
            varVDn = self.sphericalModel(self.vNugget, self.vRange, dzDn)
            feat.append(propDn + np.sqrt(2 * varH))
            feat.append(propDn + np.sqrt(2 * varVDn))
        return feat


    """
    This function scales the input data using min-max scaling
    Parameters:
    X - data to be scaled 
    """        
    def scaleData(self, X):
        return self.scaler.transform(X)

    """
    This method returns Euclidian distance between
    the two points p1 and p2.
    Parameters:
    p1 - first point
    p2 second point
    """
    def dist(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    """
    This function returns the input feature for the passed
    ParallelInput object.
    Parameters:
    parallelInputPC - instance of class ParallelInputPC    
    """
    def getInputFeatureForPointCloud(self, parallelInputPC):
        pi = np.array([parallelInputPC.x, parallelInputPC.y])
        dxys = []
        nWells = len(parallelInputPC.wellXY)
        for j in range(nWells):
            pj = np.array(parallelInputPC.wellXY[j])
            dxys.append(self.dist(pi, pj))
        nbrIdxs = np.argsort(dxys)
        nbrIdxs = nbrIdxs[0:parallelInputPC.nNeighborWells]
        feat = []
        for i in nbrIdxs:            
            prop, zEff = self.calcPropValue(parallelInputPC.z, i)
            feat.append(prop)
            
            dxy = dxys[i]
            varH = self.sphericalModel(parallelInputPC.hNugget, 
                                                parallelInputPC.hRange, dxy)
            dz = abs(parallelInputPC.z - zEff)
            varV = self.sphericalModel(parallelInputPC.vNugget, 
                                                parallelInputPC.vRange, dz)
            feat.append(prop + np.sqrt(2 * varH))
            feat.append(prop + np.sqrt(2 * varV))
            
            propUp, zEffUp = self.calcPropValue(parallelInputPC.z + 
                                                         parallelInputPC.zShift, i)
            feat.append(propUp)
            dzUp = abs(parallelInputPC.z - zEffUp)
            varVUp = self.sphericalModel(parallelInputPC.vNugget, 
                                                  parallelInputPC.vRange, dzUp)
            feat.append(propUp + np.sqrt(2 * varH))
            feat.append(propUp + np.sqrt(2 * varVUp))
            
            propDn, zEffDn = self.calcPropValue(parallelInputPC.z - 
                                                         parallelInputPC.zShift, i)
            feat.append(propDn)
            dzDn = abs(parallelInputPC.z - zEffDn)
            varVDn = self.sphericalModel(parallelInputPC.vNugget, 
                                                  parallelInputPC.vRange, dzDn)
            feat.append(propDn + np.sqrt(2 * varH))
            feat.append(propDn + np.sqrt(2 * varVDn))
        return feat


    def formulatePointCloudInput(self, pointCloudPath):
        # Reading the point cloud
        pc = pd.read_csv(pointCloudPath)
        xCol = pc.columns.get_loc(self.xColName)
        yCol = pc.columns.get_loc(self.yColName)
        zCol = pc.columns.get_loc(self.zColName)
        xpc = pc.iloc[:, xCol].values.reshape(-1, 1).astype("float32")
        ypc = pc.iloc[:, yCol].values.reshape(-1, 1).astype("float32")
        zpc = pc.iloc[:, zCol].values.reshape(-1, 1).astype("float32")
        nSamples = xpc.shape[0]
        nanIdxs = np.where(np.logical_or(np.logical_or(xpc == self.xyzNDV, 
                                                        ypc == self.xyzNDV), 
                                                        zpc == self.xyzNDV))[0]
        
        features = []
        for i in range(nSamples):
            parallelInputPC = ParallelInputPC(xpc[i][0], ypc[i][0], zpc[i][0], 
                                                self.zShift, self.wellXY, 
                                                self.wellZMin, self.wellZMax, 
                                                self.grnns, self.wellProp[:][0], 
                                                self.wellProp[:][-1], self.nNeighborWells,
                                                self.sill, self.vRange, self.hRange, 
                                                self.vNugget, self.hNugget)
            features.append(self.getInputFeatureForPointCloud(parallelInputPC))
        features = np.array(features).astype("float32")
        idxs = np.where(np.logical_or(np.isnan(features), np.isinf(features)))
        features[idxs] = self.propMean
        return features, nanIdxs

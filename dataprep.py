import os
import tensorflow as tf
import numpy as np
from B4RRegression import B4RRegressor

from sklearn.model_selection import train_test_split


class DataPrep():
    def __init__(self, data_dir="", xColName="X", yColName="Y", zColName="Z", propColName="prop1",
                    wellColName="wellID", sill=0.004238, hNugget=0.003224, hRange=4974.917187,
                    vNugget=0.003224, vRange=0.284656, nNeighborWells=6):
        
        
        # running with the pipelineAI CLI
        self.inputFile = "/input_train.csv"
        self.inputPath = data_dir + self.inputFile
        print ("*** input file path is :: " + self.inputPath)

        """
        # running without the pipelineAI CLI
        #data_dir = "/Users/arnab/devwork/lgcwork/basicDNN/model/
        self.data_dir = os.path.dirname(data_dir)
        self.inputFile = "/input/input_train.csv"
        self.inputPath = self.data_dir + self.inputFile
        """

        self.xColName = xColName
        self.yColName = yColName
        self.zColName = zColName
        self.propColName = propColName
        self.wellColName = wellColName
        self.sill = sill
        self.hNugget = hNugget
        self.hRange = hRange
        self.vNugget = vNugget
        self.vRange = vRange
        self.shiftZ = 0.075
        self.nNeighborWells = nNeighborWells
        self.validFrac = 0.2
        self.xyzNDV = -999.0
        self.propNDV = -999.0
        self.seed = 12345
        self.pointCloudFeatures = None

        self.b4rReg = B4RRegressor(self.inputPath, self.xColName, self.yColName,
                                  self.zColName, self.propColName, self.wellColName,
                                  self.seed, self.sill, self.hNugget, self.hRange,
                                  self.vNugget, self.vRange, self.shiftZ, self.nNeighborWells,
                                  self.xyzNDV, self.propNDV)

        # calcualte the feature matrix and extract the label data
        self.X, self.y = self.b4rReg.formulateInputOutput()
        self.y = self.y.reshape(-1, 1)

        # scale the data
        self.X = self.b4rReg.scaleData(self.X)

        # dont scale the labels
        #self.y = self.b4rReg.scaleData(self.y)

        """
        # print all the numpy arrays computed inside self.b4rReg
        print ("\n\n\n")
        print ("*** self.wells ***")
        print (self.b4rReg.wells.reshape(1,-1))
        print ()

        print ("*** self.uniqueWells ***")
        print (self.b4rReg.uniqueWells)
        print ()

        print ("*** self.nPointsWell ***")
        print (self.b4rReg.nPointsWell)
        print ()

        print ("*** self.wellXY ***")
        print (self.b4rReg.wellXY)
        print ()

        print ("*** self.wellZ ***")
        print (self.b4rReg.wellZ)
        print ()

        print ("*** self.wellZMin ***")
        print (self.b4rReg.wellZMin)
        print ()

        print ("*** self.wellZMax ***")
        print (self.b4rReg.wellZMax)
        print ()

        print ("*** self.wellProp ***")
        print (self.b4rReg.wellProp)
        print ()

        print ("*** self.wellNeighbors ***")
        print (self.b4rReg.wellNeighbors)
        print ()

        print ("*** self.wellPairsDist ***")
        print (self.b4rReg.wellPairsDist)
        print ()
        print ("\n\n\n")
        """

        # split the data into train and validation sets
        #self.X, self.xHoldOut, self.y, self.yHoldOut = train_test_split(self.X, self.y, test_size=self.validFrac, random_state=self.seed)
        self.xTrain, self.xValid, self.yTrain, self.yValid = train_test_split(self.X, self.y, test_size=self.validFrac, random_state=self.seed)

        # print some assessment information
        print ("Shape of xTrain = " + str(self.xTrain.shape))
        print ("Shape of xValid = " + str(self.xValid.shape))

    # create a TF Dataset from the training data
    def train(self):
        return tf.data.Dataset.from_tensor_slices((self.xTrain, self.yTrain))
    
    # create a TF Dataset from the validation data
    def validate(self):
        return tf.data.Dataset.from_tensor_slices((self.xValid, self.yValid))

    # create a TF Dataset from the pointcloud data for prediction step
    def predict(self):
        return tf.data.Dataset.from_tensor_slices(self.pointCloudFeatures)

    # pre-process the pointcloud data to calculate features
    def processPointCloudData(self, pointCloudPath):
        self.pointCloudFeatures, nanIdxs = self.b4rReg.formulatePointCloudInput(pointCloudPath)
        self.pointCloudFeatures = self.b4rReg.scaleData(self.pointCloudFeatures)
        return nanIdxs

if __name__ == '__main__':
    dp = DataPrep()
    print (" !!! FINISH !!! ")
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:10:44 2017

@author: H128815 <yogendra.pandey@halliburton.com>
BRAINS4Reservoirs - Deep Learning for Static Reservoir Modeling.
"""
import sys
# The following line adds required modules to PYTHONPATH
sys.path.insert(0, '/mnt/batch/tasks/shared/LS_root/mounts/mynfs/B4R/regression.zip')

from sklearn.model_selection import train_test_split
import regression.B4RRegression as reg
import tensorflow as tf
from tensorflow.python.client import device_lib

import os
import time

def getFlags():
    flags = tf.app.flags
    flags.DEFINE_string("data_dir", "/tmp/data",
                        "Directory for storing input data")
    flags.DEFINE_string("output_dir", "/tmp/output",
                        "Directory for storing output data")
    flags.DEFINE_integer("task_index", None,
                         "Worker task index, should be >= 0. task_index=0 is "
                         "the master worker task the performs the variable "
                         "initialization ")
    flags.DEFINE_integer("num_gpus", 1,
                         "Total number of gpus for each machine."
                         "If you don't use GPU, please set it to '0'")
    flags.DEFINE_integer("replicas_to_aggregate", None,
                         "Number of replicas to aggregate before parameter update"
                         "is applied (For sync_replicas mode only; default: "
                         "num_workers)")
    flags.DEFINE_boolean("sync_replicas", False,
                         "Use the sync_replicas (synchronized replicas) mode, "
                         "wherein the parameter updates from workers are aggregated "
                         "before applied to avoid stale gradients")
    flags.DEFINE_boolean("existing_servers", False, "Whether servers already exists. If True, "
                         "will use the worker hosts via their GRPC URLs (one client process "
                         "per worker host). Otherwise, will create an in-process TensorFlow "
                         "server.")
    flags.DEFINE_string("ps_hosts","localhost:2222",
                        "Comma-separated list of hostname:port pairs")
    flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                        "Comma-separated list of hostname:port pairs")
    flags.DEFINE_string("job_name", None,"job name: worker or ps")
    flags.DEFINE_float("sill", 0.00282744,"Variogram sill")
    # lines hNugget, hRange, vNugget,vRange, nNeighborWells, inputFile, xColName, yColName, zColName,
    # propColName, wellColName are user supplied
    flags.DEFINE_float("hNugget", 0.00017561,"Nugget for horizontal variogram")
    flags.DEFINE_float("hRange", 556.81818182,"Range for horizontal variogram")
    flags.DEFINE_float("vNugget", 0.00093717,"Nugget for vertical variogram")
    flags.DEFINE_float("vRange", 12.25836204,"Range for vertical variogram")
    flags.DEFINE_integer("nNeighborWells", 8,"Number of neighbor wells for input feature formation")
    flags.DEFINE_float("shiftZ", 0.075,"Fraction of vertical variogram range to shift by "
                       "for generating additional points at neighboring wells")
    flags.DEFINE_string("inputFile", "input.csv", "Name of input file")
    flags.DEFINE_string("xColName", "X", "x-coordinate column name in input file")
    flags.DEFINE_string("yColName", "Y", "y-coordinate column name in input file")
    flags.DEFINE_string("zColName", "Z", "z-coordinate column name in input file")
    flags.DEFINE_string("propColName", "POR", "Input property column name in input file")
    flags.DEFINE_string("wellColName", "WELL", "Well ID column name in input file")
    flags.DEFINE_integer("nNodes", 72, "Number of nodes in each hidden layer")
    flags.DEFINE_integer("nLayers", 3, "Number of hidden layers in the neural network")
    flags.DEFINE_integer("batchSize", 352, "Batch size for stochastic optimization")
    flags.DEFINE_integer("seed", 123456, "Random seed for numpy and TensorFlow")
    flags.DEFINE_integer("nSteps", 30001, "Number of iterations/steps for neural network training")
    flags.DEFINE_integer("logSteps", 1000, "Number of iterations/steps for logging during training")
    flags.DEFINE_float("stdDev", 0.1,"Standard deviation for weights initialization")
    flags.DEFINE_float("regParam", 0.0000525,"Regularization parameter for neural network training")
    flags.DEFINE_float("initRate", 0.0000125,"Initial learning rate for neural network training")
    flags.DEFINE_float("validFrac", 0.1667,"Fraction of data to be allocated for cross-validation")
    # user supplied data: xyzNDV, propNDV
    flags.DEFINE_float("xyzNDV", -999.0,"No data value for x, y, and z coordinates")
    flags.DEFINE_float("propNDV", -999.25,"No data value for the property")
    flags.DEFINE_string("objFunValFile", None, "File containing objective function value "
                        "for current run (absolute path)")
    flags.DEFINE_boolean("azureFlag", False, "Flag indicating whether to run on Azure batch")
    flags.DEFINE_boolean("useGPUs", False, "Flag indicating whether to use GPUs")
    flags.DEFINE_boolean("verbose", True, "Flag indicating whether to print detailed information")
    return flags.FLAGS

# Track execution time
startTime = time.time()
flags = getFlags()
azureFlag = flags.azureFlag
useGPUs = flags.useGPUs
verbose = flags.verbose
batchSize = flags.batchSize
nFeat = flags.nNeighborWells * 9
nSteps = flags.nSteps
nLayers = flags.nLayers
nNodes = []
for i in range(nLayers):
    nNodes.append(flags.nNodes)
seed = flags.seed
logSteps = flags.logSteps
regParam = flags.regParam
initRate = flags.initRate
stdDev = flags.stdDev
validFrac = flags.validFrac
inputPath = flags.inputFile
if azureFlag:
    mountRoot = os.environ['AZ_LEARNING_MOUNT_ROOT']
    inputPath = mountRoot + '/mynfs/B4R/data/' + inputPath
else:
    dataFolder = os.path.join(os.getcwd(), 'data')
    inputPath = os.path.join(dataFolder, inputPath)
    
propColName = flags.propColName
checkpointFile = 'regressor.' + propColName + '.ckpt'
if verbose:
    print('Checkpoint file is: ' + checkpointFile)
objFunValFile = flags.objFunValFile
modelFolder = os.path.join(os.getcwd(), 'model')
if azureFlag:
    modelFolder = os.environ['AZ_LEARNING_OUTPUT_modelOutput']
if not os.path.exists(modelFolder):
    os.makedirs(modelFolder)
checkpointFilePath = os.path.join(modelFolder, checkpointFile)
if verbose:
    print('Checkpoint file path is: ' + checkpointFilePath)
objFunValFilePath = None
if objFunValFile != None:
    objFunValFilePath = os.path.join(modelFolder, objFunValFile)

psSpec = flags.ps_hosts.split(",")
workerSpec = flags.worker_hosts.split(",")
# Preparing the train, test, and validation data sets
b4rReg = reg.B4RRegressor(inputPath, flags.xColName, flags.yColName, 
                          flags.zColName, propColName, flags.wellColName, 
                          seed, flags.sill, flags.hNugget, flags.hRange, 
                          flags.vNugget, flags.vRange, psSpec, workerSpec, 
                          flags.task_index, flags.shiftZ, flags.existing_servers, 
                          flags.job_name, flags.num_gpus, flags.nNeighborWells,
                          flags.xyzNDV, flags.propNDV)
X, y, scaler = b4rReg.formulateInputOutput()
y = y.reshape(-1, 1)
X, y = b4rReg.scaleData(X, y, scaler)
y = y.reshape(-1, 1)

# split the data into validation and test sets
X, xHoldOut, y, yHoldOut = train_test_split(X, y, validFrac, random_state=seed)
xTrain, xValid, yTrain, yValid = train_test_split(X, y, validFrac, random_state=seed)

# Turn off information logging for TensorFlow
if verbose != True:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = None
if useGPUs:
    local_devices = device_lib.list_local_devices()
    gpus = [x.name for x in local_devices if x.device_type == 'GPU']
    if verbose:
        print('Available GPUs: %s' % gpus)
# Train the deep neural network
tf.set_random_seed(seed)
graphOption = None
graph = None
if useGPUs:
    graphOption = tf.device(gpus[0])
else:
    graph = tf.Graph()
    graphOption = graph.as_default()
with graphOption:
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tfTrainData = tf.placeholder(tf.float32, shape=(batchSize, nFeat))
    tfTrainProp = tf.placeholder(tf.float32, shape=(batchSize,))
    tfValidData = tf.constant(xValid, dtype=tf.float32)
    tfTestData = tf.constant(xHoldOut, dtype=tf.float32)
    tfAllData = tf.constant(X, dtype=tf.float32)

    logits, w, b, wo = b4rReg.train(tfTrainData, nNodes, stdDev, nLayers, nFeat)
    loss = 0.5 * tf.reduce_mean(tf.square(tf.transpose(logits) - tfTrainProp))
    # Compute regularization term
    regTerm = tf.Variable(0.0)
    for i in range(nLayers):
        regTerm = regTerm + tf.reduce_mean(tf.nn.l2_loss(w[i]))
    regTerm = regTerm * regParam
    # Add regularization term to loss
    loss = loss + regTerm
    
    # Optimizer.
    # Exponential decay of learning rate.
    # count the number of steps taken.
    globalStep = tf.Variable(0.0, name="globalStep", trainable=False)
    learningRate = tf.train.exponential_decay(initRate, globalStep, 2000, 0.86, staircase=True)
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    trainPred = logits
    validPred = b4rReg.predict(tfValidData, w, b, wo)
    testPred = b4rReg.predict(tfTestData, w, b, wo)
    allPred = b4rReg.predict(tfAllData, w, b, wo)

sessionOption = None
if useGPUs:
    sessionOption = tf.Session(config=tf.ConfigProto(log_device_placement=True))
else:
    sessionOption = tf.Session(graph=graph)
with sessionOption as session:
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    if verbose:
        print("Initialized TensorFlow session...")
    for step in range(nSteps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batchSize) % (yTrain.shape[0] - batchSize)
        # Generate a minibatch.
        batchData = xTrain[offset:(offset + batchSize), :]
        batchProp = yTrain[offset:(offset + batchSize)]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feedDict = {tfTrainData : batchData, tfTrainProp : batchProp}
        _, l, pred = session.run([optimizer, loss, trainPred], feed_dict=feedDict)
        if (step % logSteps == 0):
            if verbose:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch RMSE: %f" % b4rReg.rmse(pred, batchProp))
                validRMSE = b4rReg.rmse(validPred.eval(), yValid)
                print("Validation RMSE: %f" % validRMSE)
            if step == (nSteps - 1):
                saver.save(session, checkpointFilePath)
    print("Test RMSE: %f" % b4rReg.rmse(testPred.eval(), yHoldOut))
    print("Test Corr. Coeff.: %f" % b4rReg.claculateCorrCoef(testPred.eval(), yHoldOut))
    rmseAll = b4rReg.rmse(allPred.eval(), y)
    print("Total RMSE: %f" % rmseAll)
    corrAll = b4rReg.claculateCorrCoef(allPred.eval(), y)
    print("Total Corr. Coeff.: %f" % corrAll)
    if objFunValFile != None:
        f = open(objFunValFilePath, 'w')
        f.write(str(rmseAll) + ' ' + str(corrAll))
        f.close()
endTime = time.time()
print("Total time elapsed: %f s" % (endTime - startTime))


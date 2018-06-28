from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import itertools

import tensorflow as tf
import pandas as pd
import numpy as np
from dataprep import DataPrep

def distribution_strategy(num_gpus):

    if num_gpus == 1:
        return tf.contrib.distribute.OneDeviceStrategy(device='/GPU:0')
    elif num_gpus > 1:
        return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
    else:
        return None


class Model(object):
    """Class that defines the deep NN architecture"""

    def __init__(self, nLayers, nUnits):
        self.nLayers = nLayers
        self.nUnits = nUnits

    def __call__(self, inputs):
        y = inputs
        for _ in range(self.nLayers):
            y = tf.layers.dense(inputs=y, units=self.nUnits, activation=tf.nn.relu)
        return tf.layers.dense(inputs=y, units=1, activation=tf.nn.relu)


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    model = Model(params['nLayers'], params['nUnits'])
    X = features
    if isinstance(X, dict):
        X = features['inputs']

    print('**MODE: %s**' % mode)
    print('**MODE: %s**' % mode)

    # Train mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = model(X)
        loss = 0.5 * tf.reduce_mean(tf.square(logits - labels))
        MSE = tf.metrics.mean_squared_error(labels=labels, predictions=logits)
        # Name the MSE tensor 'train_MSE' to demonstrate the LoggingTensorHook.
        tf.identity(MSE[1], name='train_MSE')
        tf.summary.scalar('train_MSE', MSE[1])
        optimizer = tf.train.AdamOptimizer(learning_rate=params['initRate'])
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)        
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))


def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    if not num_gpus:
        raise ValueError('Multi-GPU mode was specified, but no GPUs '
                         'were found. To use CPU, run without --multi_gpu.')

    remainder = batch_size % num_gpus
    if remainder:
        err = ('When running with multiple GPUs, batch size '
               'must be a multiple of the number of available GPUs. '
               'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
               ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)


def main(unused_argv):

    # define a DataPrep object
    dp = DataPrep(FLAGS.data_dir, FLAGS.xColName, FLAGS.yColName, FLAGS.zColName,
                    FLAGS.propColName, FLAGS.wellColName, FLAGS.sill,
                    FLAGS.hNugget, FLAGS.hRange, FLAGS.vNugget, FLAGS.vRange,
                    FLAGS.nNeighborWells)

    model_function = model_fn
    run_config = tf.estimator.RunConfig(train_distribute=distribution_strategy(1))

    # define the predictor estimator
    basicDDN_predictor = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=FLAGS.model_dir,
        params={
            'nLayers': FLAGS.nLayers,
            'nUnits': FLAGS.nUnits,
            'initRate': FLAGS.initRate,
            'batch_size': FLAGS.batch_size
        },
        config=run_config
    )

    # Train the model
    def train_input_fn():
        ds = dp.train()
        ds = ds.shuffle(buffer_size=50000)
        ds = ds.cache().repeat(FLAGS.train_epochs).batch(FLAGS.batch_size)
        ds = ds.prefetch(1)
        return ds

    # Set up training hook that logs the training MSE every 100 steps.
    tensors_to_log = {'train_MSE': 'train_MSE'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    
    # Train the model in a distributed fashion
    basicDDN_predictor.train(input_fn=train_input_fn, hooks=[logging_hook])


class BasicDNNArgParser(argparse.ArgumentParser):

    def __init__(self):
        super(BasicDNNArgParser, self).__init__()

        """
        self.add_argument(
            '--multi_gpu', action='store_true',
            help='If set, run across all available GPUs.')
        """
        self.add_argument(
            '--batch_size',
            type=int,
            default=100,
            help='Number of examples to process in a batch (ie. 400)')
        self.add_argument(
            '--data_dir',
            type=str,
            #default=os.path.dirname(os.path.abspath(__file__)),
            default=os.environ['PIPELINE_INPUT_PATH'],
            help='Path to directory containing the training dataset')
        self.add_argument(
            '--input_pointcloud_file',
            type=str,
            #default='/Users/arnab/devwork/lgcwork/basicDNN/input/input_pcfile.csv',
            default='%s/input_pcfile.csv' % os.environ['PIPELINE_INPUT_PATH'],
            help='Path of the input pointcloud file')
        self.add_argument(
            '--output_pointcloud_file',
            type=str,
            #default='/Users/arnab/devwork/lgcwork/basicDNN/input/output_pcfile.csv',
            default='%s/output_pcfile.csv' % os.environ['PIPELINE_INPUT_PATH'],
            help='Path of the output file to write the predictions on the pointcloud data')
        self.add_argument(
            '--intermediate_pointcloud_file',
            type=str,
            #default='/Users/arnab/devwork/lgcwork/basicDNN/input/int_pcfile.csv',
            default='%s/features_pcfile.csv' % os.environ['PIPELINE_INPUT_PATH'],
            help='Path of the intermediate file to write the computed point cloud features')
        self.add_argument(
            '--model_dir',
            type=str,
            #default='../output/training',
            default='%s/output/training' % os.environ['PIPELINE_OUTPUT_PATH'],
            help='The directory where the model checkpoints during training will be stored.')
        self.add_argument(
            '--train_epochs',
            type=int,
            default=10,
            help='Number of epochs to train (ie. 20).')
        """
        self.add_argument(
            '--data_format',
            type=str,
            default=None,
            choices=['channels_first', 'channels_last'],
            help='A flag to override the data format used in the model. '
            'channels_first provides a performance boost on GPU but is not always '
            'compatible with CPU. If left unspecified, the data format will be '
            'chosen automatically based on whether TensorFlow was built for CPU or '
            'GPU.')
        """
        self.add_argument(
            '--export_dir',
            type=str,
            #default='./pipeline_tfserving/0',
            default='%s/pipeline_tfserving' % os.environ['PIPELINE_OUTPUT_PATH'],
            help='The directory where the exported SavedModel will be stored.')
        self.add_argument(
            '--sill',
            type=float,
            default=0.004238,
            help='sill value')
        self.add_argument(
            '--vNugget',
            type=float,
            default=0.003224,
            help='vertical nugget value')
        self.add_argument(
            '--hNugget',
            type=float,
            default=0.003224,
            help='horizontal nugget value')
        self.add_argument(
            '--vRange',
            type=float,
            default=0.284656,
            help='vertical range value')
        self.add_argument(
            '--hRange',
            type=float,
            default=4974.917187,
            help='horizontal range value')
        self.add_argument(
            '--initRate',
            type=float,
            default=0.0001,
            help='learning rate')
        self.add_argument(
            '--nUnits',
            type=int,
            default=10,
            help='number of units in each hidden layer')
        self.add_argument(
            '--nLayers',
            type=int,
            default=4,
            help='number of layers in the NN')
        self.add_argument(
            '--nNeighborWells',
            type=int,
            default=6,
            help='number of neighboring wells')
        self.add_argument(
            '--wellColName',
            type=str,
            default='WELL',
            help='the column name of the well ID')
        self.add_argument(
            '--xColName',
            type=str,
            default='X',
            help='The column name for the X location')
        self.add_argument(
            '--yColName',
            type=str,
            default='Y',
            help='The column name for the Y location')
        self.add_argument(
            '--zColName',
            type=str,
            default='Z',
            help='The column name for the Z location')
        self.add_argument(
            '--propColName',
            type=str,
            default='POR',
            help='The property name in the dataset for which the model is being trained')


if __name__ == '__main__':
    parser = BasicDNNArgParser()
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

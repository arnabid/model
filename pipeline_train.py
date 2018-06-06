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

    # Inference mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(X)
        predictions = {
            'predictions': logits,
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Train mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['initRate'])
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

        logits = model(X)
        loss = 0.5 * tf.reduce_mean(tf.square(logits - labels))
        MSE = tf.metrics.mean_squared_error(labels=labels, predictions=logits)
        # Name the MSE tensor 'train_MSE' to demonstrate the LoggingTensorHook.
        tf.identity(MSE[1], name='train_MSE')
        tf.summary.scalar('train_MSE', MSE[1])
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    # Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(X)
        loss = 0.5 * tf.reduce_mean(tf.square(logits - labels))
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'MSE':
                    tf.metrics.mean_squared_error(
                        labels=labels,
                        predictions=logits),
            })


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
    model_function = model_fn

    """
    if FLAGS.multi_gpu:
        validate_batch_size_for_multi_gpu(FLAGS.batch_size)

        # There are two steps required if using multi-GPU: (1) wrap the model_fn,
        # and (2) wrap the optimizer. The first happens here, and (2) happens
        # in the model_fn itself when the optimizer is defined.
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_fn, loss_reduction=tf.losses.Reduction.MEAN)

    data_format = FLAGS.data_format
    if data_format is None:
        data_format = ('channels_first'
                       if tf.test.is_built_with_cuda() else 'channels_last')
    """

    # run_config=tf.estimator.RunConfig(model_dir=os.path.join(os.environ['PIPELINE_OUTPUT_PATH'],
    #                                                            'pipeline_tfserving/0')),

    # define a DataPrep object
    dp = DataPrep(FLAGS.data_dir, FLAGS.xColName, FLAGS.yColName, FLAGS.zColName,
                    FLAGS.propColName, FLAGS.wellColName, FLAGS.sill,
                    FLAGS.hNugget, FLAGS.hRange, FLAGS.vNugget, FLAGS.vRange,
                    FLAGS.nNeighborWells)

    # define the predictor estimator
    petroDDN_predictor = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=FLAGS.model_dir,
        params={
            'nLayers': FLAGS.nLayers,
            'nUnits': FLAGS.nUnits,
            'initRate': FLAGS.initRate,
            'batch_size': FLAGS.batch_size
        },
    )
    #    config=run_config)

    # Train the model
    def train_input_fn():
        ds = dp.train()
        ds = ds.cache().batch(FLAGS.batch_size).repeat(FLAGS.train_epochs)
        ds = ds.shuffle(buffer_size=50000)
        
        # Return the next batch of data.
        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels

    # Set up training hook that logs the training MSE every 100 steps.
    tensors_to_log = {'train_MSE': 'train_MSE'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    
    # Train the model
    petroDDN_predictor.train(input_fn=train_input_fn, hooks=[logging_hook])

    # Evaluate the model and print results
    def eval_input_fn():
        return dp.validate().batch(FLAGS.batch_size).make_one_shot_iterator().get_next()

    eval_results = petroDDN_predictor.evaluate(input_fn=eval_input_fn)
    print()
    print('Evaluation results:\n\t%s' % eval_results)

    # Export the model
    if FLAGS.export_dir is not None:
        X = tf.placeholder(tf.float32, shape=(
            FLAGS.batch_size, FLAGS.nNeighborWells * 9))
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'inputs': X,
        })
        petroDDN_predictor.export_savedmodel(FLAGS.export_dir, input_fn)

    # Predict property values at locations in the pointcloud file
    # indices where there is missing data
    nanIdxs = dp.processPointCloudData(FLAGS.input_pointcloud_file)

    def predict_input_fn():
        return dp.predict().batch(FLAGS.batch_size).make_one_shot_iterator().get_next(), None
    
    predictions = petroDDN_predictor.predict(input_fn=predict_input_fn)
    values = np.array(list(map(lambda item: item["predictions"][0],list(itertools.islice(predictions, 0, None)))))
    #values = values * (dp.b4rReg.propMax - dp.b4rReg.propMin) + dp.b4rReg.propMin
    values[nanIdxs] = dp.propNDV
    #print('\n\nPrediction results:\n\t%s' % values)

    # write the computed pointcloud features to an intermediate file
    op_in = pd.DataFrame(data=dp.pointCloudFeatures)
    op_in.to_csv(FLAGS.intermediate_pointcloud_file, index=None)

    # write the predictions to the output pointcloud file
    op = pd.DataFrame(data=values, columns=[FLAGS.propColName])
    op.to_csv(FLAGS.output_pointcloud_file, index=None)


class PetroDNNArgParser(argparse.ArgumentParser):

    def __init__(self):
        super(PetroDNNArgParser, self).__init__()

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
            default='%s/int_pcfile.csv' % os.environ['PIPELINE_INPUT_PATH'],
            help='Path of the intermediate file to write the point cloud features')
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
    parser = PetroDNNArgParser()
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

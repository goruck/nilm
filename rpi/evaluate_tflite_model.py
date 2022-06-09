"""
Evaluate tflite model performance on raspberry pi.

Copyright (c) 2022 Lindo St. Angel.
"""

import os
import argparse
import socket
import sys

import tflite_runtime.interpreter as tflite
import numpy as np

sys.path.append('../ml')
import nilm_metric as nm
from logger import log
import common

# Dataset energy sampling period in seconds. 
SAMPLE_PERIOD = 8

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Evaluate a tflite model on the raspberry pi.')
    parser.add_argument('--appliance_name',
        type=str,
        default='kettle',
        help='Name of target appliance.')
    parser.add_argument('--datadir',
        type=str,
        default='/home/pi/nilm/ml/dataset_management/refit/',
        help='Directory of datasets.')
    parser.add_argument('--model_path',
        type=str,
        default='/home/pi/nilm/ml/models/',
        help='tflite model path')
    parser.add_argument('--crop',
        type=int,
        default=None,
        help='Number of dataset samples to use. Default uses entire dataset.')
    parser.add_argument('--num_eval',
        type=int,
        default=100000,
        help='Number of inferences used to evaluate quantized model.')
    parser.add_argument('--test_type',
        type=str,
        default='test',
        help='Type of the test set to load: \
            test -- test on the proper test set;\
            train -- test on a already prepared slice of the train set;\
            val -- test on the validation set;\
            uk -- test on UK-DALE;\
            redd -- test on REDD.')
    return parser.parse_args()

if __name__ == '__main__':
    log(f'Machine name: {socket.gethostname()}')
    args = get_arguments()
    log('Arguments: ')
    log(args)

    # The appliance to train on.
    appliance_name = args.appliance_name

    # Load tflite model.
    model_filepath = os.path.join(args.model_path, appliance_name,
        f'{appliance_name}_quant.tflite')
    log(f'tflite model: {model_filepath}')

    # Load dataset.
    test_file_name = common.find_test_filename(
        args.datadir, appliance_name, args.test_type)
    dataset_path = os.path.join(args.datadir, appliance_name, test_file_name)
    log(f'dataset: {dataset_path}')
    dataset = common.load_dataset(dataset_path, args.crop)
    num_samples = dataset[0].size
    log(f'Loaded {num_samples/10**6:.3f}M samples from dataset.')

    # Provider of windowed dataset samples and single point targets.
    WindowGenerator = common.get_window_generator(keras_sequence=False)
    provider = WindowGenerator(
        dataset=dataset,
        batch_size=1,
        shuffle=False)

    # Perform inference on windowed samples.
    interpreter = tflite.Interpreter(model_path=model_filepath)
    results = common.tflite_infer(
        interpreter=interpreter,
        provider=provider,
        num_eval=args.num_eval,
        log=log)

    ground_truth = np.array([g for g, _ in results])
    prediction = np.array([p for _, p in results])

    # De-normalize.
    appliance_mean = common.params_appliance[appliance_name]['mean']
    log(f'appliance mean: {appliance_mean}')
    appliance_std = common.params_appliance[appliance_name]['std']
    log(f'appliance std: {appliance_std}')
    prediction = prediction * appliance_std + appliance_mean
    ground_truth = ground_truth * appliance_std + appliance_mean

    # Apply on-power threshold.
    appliance_threshold = common.params_appliance[
        appliance_name]['on_power_threshold']
    log(f'appliance threshold: {appliance_threshold}')
    prediction[prediction < appliance_threshold] = 0.0
    
    # Calculate absolute error statistics. 
    (mean, std, min, max, quartile1,
    median, quartile2, _) = nm.get_abs_error(ground_truth, prediction)
    log(f'abs error - mean: {mean} std: {std} \n'
        f'min: {min} max: {max} quartile1: {quartile1} \n'
        f'median: {median} quartile2: {quartile2}')

    # Calculate normalized disaggregation error.
    log(f'nde: {nm.get_nde(ground_truth, prediction)}')
    
    # Calculate normalized signal aggregate error. 
    log(f'sae: {nm.get_sae(ground_truth, prediction, SAMPLE_PERIOD)}')
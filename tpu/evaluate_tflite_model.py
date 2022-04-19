"""
Evaluate tflite model on the edge tpu.

Copyright (c) 2022 Lindo St. Angel.
"""

import os
import argparse
import socket
import time

import tflite_runtime.interpreter as tflite
import numpy as np

import nilm_metric as nm
import common
from logger import log

# Number of samples for post-training quantization calibration.
NUM_CAL = 1000
# Dataset energy sampling period in seconds. 
SAMPLE_PERIOD = 8

rng = np.random.default_rng()

def tflite_infer(model, provider, num_eval) -> list:
    """Perform inference using a tflite model."""

    # Start the tflite interpreter on the tpu and allocate tensors.
    interpreter = tflite.Interpreter(model=model,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    log(f'interpreter input details: {input_details}')
    output_details = interpreter.get_output_details()
    log(f'interpreter output details: {output_details}')
    # Check I/O tensor type.
    floating_input = input_details[0]['dtype'] == np.float32
    log(f'tflite model floating input: {floating_input}')
    floating_output = output_details[0]['dtype'] == np.float32
    log(f'tflite model floating output: {floating_output}')
    # Get I/O indices.
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    # If model has int I/O get quantization information.
    if not floating_input:
        input_scale, input_zero_point = input_details[0]['quantization']
    if not floating_output:
        output_scale, output_zero_point = output_details[0]['quantization']

    # Calculate num_eval sized indices of random locations in provider.
    eval_indices = rng.integers(low=0, high=provider.__len__(), size=num_eval)

    log(f'Running inference on {num_eval} samples...')
    start = time.time()
    def infer(i):
        sample, target = provider.__getitem__(i)
        if not sample.any(): return # ignore missing data
        ground_truth = np.squeeze(target)
        if not floating_input: # convert to float to int8
            sample = sample / input_scale + input_zero_point
            sample = sample.astype(np.int8)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke() # run inference
        result = interpreter.get_tensor(output_index)
        prediction = np.squeeze(result)
        if not floating_output: # convert int8 to float
            prediction = (prediction - output_zero_point) * output_scale
        #print(f'sample index: {i} ground_truth: {ground_truth:.3f} prediction: {prediction:.3f}')
        return ground_truth, prediction
    results = [infer(i) for i in eval_indices]
    end = time.time()
    log('Inference run complete.')
    log(f'Inference rate: {num_eval / (end - start):.3f} Hz')

    return results

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Convert a trained Keras model to tflite and quantize.')
    parser.add_argument('--appliance_name',
                        type=str,
                        default='kettle',
                        help='Name of target appliance.')
    parser.add_argument('--datadir',
                        type=str,
                        default='/home/lindo/Develop/nilm/ml/dataset_management/refit',
                        help='Directory of datasets.')
    parser.add_argument('--model_path',
                        type=str,
                        default='/media/mendel/nilm/ml/models/fridge/fridge_quant_edgetpu.tflite',
                        help='tflite model path')
    parser.add_argument('--crop',
                        type=int,
                        default=None,
                        help='Number of dataset samples to use. Default uses entire dataset.')
    parser.add_argument('--io_float', action='store_true',
                        help='If set, make tflite I/O float.')
    parser.add_argument('--evaluate', action='store_true',
                        help='If set, evaluate tflite model for accuracy.')
    parser.add_argument('--num_eval',
                        type=int,
                        default=10000,
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
    parser.set_defaults(io_float=False)
    parser.set_defaults(evaluate=False)
    return parser.parse_args()

if __name__ == '__main__':
    log(f'Machine name: {socket.gethostname()}')
    args = get_arguments()
    log('Arguments: ')
    log(args)

    # The appliance to train on.
    appliance_name = args.appliance_name

    # Get tflite model path.
    model_filepath = os.path.join(args.model_path)
    log(f'Model file path: {model_filepath}')

    # Calculate offset parameter from window length.
    window_length = common.params_appliance[appliance_name]['windowlength']
    offset = int(0.5 * (window_length - 1.0))

    # Load dataset.
    test_file_name = common.find_test_filename(args.datadir, appliance_name, args.test_type)
    dataset_path = os.path.join(args.datadir, appliance_name, test_file_name)
    log(f'dataset: {dataset_path}')
    dataset = common.load_dataset(dataset_path, args.crop)
    num_samples = dataset[0].size
    log(f'Loaded {num_samples/10**6:.3f}M samples from dataset.')

    # Provider of windowed dataset samples and single point targets.
    provider = common.WindowGenerator(
        dataset=dataset,
        offset=offset,
        batch_size=1,
        shuffle=False)

    results = tflite_infer(
        model=model_filepath,
        provider=provider,
        num_eval=args.num_eval)

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
    appliance_threshold = common.params_appliance[appliance_name]['on_power_threshold']
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
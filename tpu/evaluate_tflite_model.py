"""
Evaluate tflite model performance on the edge tpu.

Copyright (c) 2022 Lindo St. Angel.
"""

import os
import argparse
import socket
import time
import sys

import tflite_runtime.interpreter as tflite
import numpy as np
import pandas as pd

sys.path.append('../ml/')
import nilm_metric as nm
from logger import log

# Dataset energy sampling period in seconds. 
SAMPLE_PERIOD = 8

rng = np.random.default_rng()

params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000},
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800},
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400},
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000},
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700}
    }

def find_test_filename(test_dir, appliance, test_type) -> str:
    """Determine test set."""
    for filename in os.listdir(os.path.join(test_dir, appliance)):
        if test_type == 'train' and 'TRAIN' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'uk' and 'UK' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'redd' and 'REDD' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'test' and 'TEST' in\
                filename.upper() and 'TRAIN' not in filename.upper() and 'UK' not in filename.upper():
            test_filename = filename
            break
        elif test_type == 'val' and 'VALIDATION' in filename.upper():
            test_filename = filename
            break
    return test_filename

def load_dataset(file_name, crop=None) -> np.array:
    """Load CSV file, convert to np and return mains and appliance samples."""
    df = pd.read_csv(file_name, nrows=crop)

    df_np = np.array(df, dtype=np.float32)

    return df_np[:, 0], df_np[:, 1]

class WindowGenerator():
    """Generate windows of samples and optionally single point targets."""

    def __init__(self, dataset, batch_size, offset,
        train=True,
        shuffle=True) -> None:

        self.X, self.y = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.offset = offset
        self.train = train

        # Total number of samples in dataset.
        self.total_samples=self.X.size

        # Number of input samples adjusted for windowing.
        # This prevents partial window generation.
        self.num_samples = self.total_samples - 2 * self.offset

        # Indices of adjusted input sample array.
        self.indices = np.arange(self.num_samples)

        self.rng = np.random.default_rng()

        # Initial shuffle. 
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def on_epoch_end(self) -> None:
        # Shuffle at end of each epoch. 
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def len(self) -> int:
        return(int(np.ceil(self.num_samples / self.batch_size)))

    def getitem(self, index) -> np.array:
        # Row indices for current batch.
        rows = self.indices[
            index * self.batch_size:(index + 1) * self.batch_size]

        # Create a batch of windowed samples.
        samples = np.array(
            [self.X[row:row + 2 * self.offset + 1] for row in rows])

        if self.train:
            # Create batch of single point targets offset from window start.
            targets = np.array([self.y[row + self.offset] for row in rows])
            # Return a batch of (sample, target) tuples.
            return samples, targets
        else:
            # Return only samples if in test mode.
            return samples

def tflite_infer(model_path, provider, num_eval) -> list:
    """Perform inference using a tflite model."""

    # Start the tflite interpreter on the tpu and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path,
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
    eval_indices = rng.integers(low=0, high=provider.len(), size=num_eval)

    log(f'Running inference on {num_eval} samples...')
    start = time.time()
    def infer(i):
        sample, target = provider.getitem(i)
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
        description='Evaluate a tflite model on the edge tpu.')
    parser.add_argument('--appliance_name',
                        type=str,
                        default='kettle',
                        help='Name of target appliance.')
    parser.add_argument('--datadir',
                        type=str,
                        default='/media/mendel/nilm/ml/dataset_management/refit/',
                        help='Directory of datasets.')
    parser.add_argument('--model_path',
                        type=str,
                        default='/media/mendel/nilm/ml/models/',
                        help='tflite model path')
    parser.add_argument('--crop',
                        type=int,
                        default=None,
                        help='Number of dataset samples to use. Default uses entire dataset.')
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
        f'{appliance_name}_quant_edgetpu.tflite')
    log(f'tflite model: {model_filepath}')

    # Calculate offset parameter from window length.
    window_length = params_appliance[appliance_name]['windowlength']
    offset = int(0.5 * (window_length - 1.0))

    # Load dataset.
    test_file_name = find_test_filename(args.datadir, appliance_name, args.test_type)
    dataset_path = os.path.join(args.datadir, appliance_name, test_file_name)
    log(f'dataset: {dataset_path}')
    dataset = load_dataset(dataset_path, args.crop)
    num_samples = dataset[0].size
    log(f'Loaded {num_samples/10**6:.3f}M samples from dataset.')

    # Provider of windowed dataset samples and single point targets.
    provider = WindowGenerator(
        dataset=dataset,
        offset=offset,
        batch_size=1,
        shuffle=False)

    results = tflite_infer(
        model_path=model_filepath,
        provider=provider,
        num_eval=args.num_eval)

    ground_truth = np.array([g for g, _ in results])
    prediction = np.array([p for _, p in results])

    # De-normalize.
    appliance_mean = params_appliance[appliance_name]['mean']
    log(f'appliance mean: {appliance_mean}')
    appliance_std = params_appliance[appliance_name]['std']
    log(f'appliance std: {appliance_std}')
    prediction = prediction * appliance_std + appliance_mean
    ground_truth = ground_truth * appliance_std + appliance_mean

    # Apply on-power threshold.
    appliance_threshold = params_appliance[appliance_name]['on_power_threshold']
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

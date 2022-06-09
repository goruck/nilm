"""
Various common modules and parameters.
"""

import os
import pandas as pd
import time

import numpy as np

params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128, },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128},
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512},
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536},
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000}
    }

def find_test_filename(test_dir, appliance, test_type) -> str:
    """Find test file name given a datset name."""
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

def get_window_generator(keras_sequence=True):
    """Wrapper to conditionally sublass WindowGenerator as Keras sequence.

    The WindowGenerator is used in keras and non-keras applications and
    so to make it useable across both it can be a subclass of a keras
    sequence. This increases reusability throughout codebase. 

    Arguments:
        keras_sequence: If true make WindowGenerator a subclass of
        the keras sequence class.
    
    Returns:
        WindowGenerator class.
    """
    if keras_sequence:
        from tensorflow import keras

    class WindowGenerator(keras.utils.Sequence if keras_sequence else object):
        """ Generates windowed timeseries samples and targets.
        
        Attributes:
            dataset: input samples, targets timeseries data.
            batch_size: mini batch size used in training model.
            window_length: number of samples in a window of timeseries data.
            train: if True returns samples and targets else just samples.
            shuffle: if True shuffles dataset initially and every epoch.
        """

        def __init__(
            self,
            dataset,
            batch_size=1000,
            window_length=599,
            train=True,
            shuffle=True) -> None:
            """Inits WindowGenerator."""

            self.X, self.y = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.window_length = window_length
            self.train = train

            # Total number of samples in dataset.
            self.total_samples=self.X.size

            # Number of samples from end of window to center.
            self.offset = int(0.5 * (window_length - 1.0))

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
            """Shuffle at end of each epoch.""" 
            if self.shuffle:
                self.rng.shuffle(self.indices)

        def __len__(self) -> int:
            """Returns number batches in an epoch."""
            return(int(np.ceil(self.num_samples / self.batch_size)))

        def __getitem__(self, index) -> np.ndarray:
            """Returns windowed samples and targets."""
            # Row indices for current batch. 
            rows = self.indices[
                index * self.batch_size:(index + 1) * self.batch_size]

            # Create a batch of windowed samples.
            samples = np.array(
                [self.X[row:row + 2 * self.offset + 1] for row in rows])

            # Reshape samples to match model's input tensor format.
            # Starting shape = (batch_size, window_length)
            # Desired shape = (batch_size, 1, window_length)
            samples = samples[:, np.newaxis, :]

            if self.train:
                # Create batch of single point targets offset from window start.
                targets = np.array([self.y[row + self.offset] for row in rows])
                return samples, targets
            else:
                # Return only samples if in test mode.
                return samples

    return WindowGenerator

def tflite_infer(interpreter, provider, num_eval, log=print) -> list:
    """Perform inference using a tflite model"""
    rng = np.random.default_rng()

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
        # Add axis to match model InputLayer shape.
        sample = sample[:, :, :, np.newaxis]
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

def normalize(dataset):
    """Normalize or standardize a dataset."""
    import numpy as np
    # Compute aggregate statistics.
    agg_mean = np.mean(dataset[0])
    agg_std = np.std(dataset[0])
    print(f'agg mean: {agg_mean}, agg std: {agg_std}')
    agg_median = np.percentile(dataset[0], 50)
    agg_quartile1 = np.percentile(dataset[0], 25)
    agg_quartile3 = np.percentile(dataset[0], 75)
    print(f'agg median: {agg_median}, agg q1: {agg_quartile1}, agg q3: {agg_quartile3}')
    # Compute appliance statistics.
    app_mean = np.mean(dataset[1])
    app_std = np.std(dataset[1])
    print(f'app mean: {app_mean}, app std: {app_std}')
    app_median = np.percentile(dataset[1], 50)
    app_quartile1 = np.percentile(dataset[1], 25)
    app_quartile3 = np.percentile(dataset[1], 75)
    print(f'app median: {app_median}, app q1: {app_quartile1}, app q3: {app_quartile3}')
    def z_norm(dataset, mean, std):
        return (dataset - mean) / std
    def robust_scaler(dataset, median, quartile1, quartile3):
        return (dataset - median) / (quartile3 - quartile1)
    return (
        z_norm(
            dataset[0], agg_mean, agg_std),
        z_norm(
            dataset[1], app_mean, app_std))
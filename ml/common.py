"""
Various common functions and parameters.
"""

import os
import pandas as pd
import time

import numpy as np

# Average of all appliance aggregate training dataset means and std's.
#AGGREGATE_MEAN = 545.0
#AGGREGATE_STD = 820.0

# Various parameters used for training, validation and testing.
params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000.0,
        'max_on_power': 3998.0,
        'train_agg_mean': 501.32453633286167,   #training aggregate mean
        'train_agg_std': 783.0367822932175,     #training aggregate standard deviation
        'train_app_mean': 16.137261776311778,   #training appliance mean
        'train_app_std': 196.89790951996966,    #training appliance standard deviation
        'test_app_mean': 23.155018918550294,    #test appliance mean
        'test_agg_mean': 465.10226795866976     #test aggregate mean
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200.0,
        'max_on_power': 3969.0,
        'train_agg_mean': 495.0447502551665,
        'train_agg_std': 704.1066664964247,
        'train_app_mean': 3.4617193220425304,
        'train_app_std': 64.22826568216946,
        'test_app_mean': 9.577146165430394,
        'test_agg_mean': 381.2162070293207
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50.0,
        'max_on_power': 3323.0,
        'train_agg_mean': 605.4483277115743,
        'train_agg_std': 952.1533235759814,
        'train_app_mean': 48.55206460642049,
        'train_app_std': 62.114631485397986,
        'test_app_mean': 24.40792692094185,
        'test_agg_mean': 254.83458540217833
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10.0,
        'max_on_power': 3964.0,
        'train_agg_mean': 606.3228537145152,
        'train_agg_std': 833.611776395652,
        'train_app_mean': 46.040618889481905,
        'train_app_std': 305.87980576285474,
        'test_app_mean': 11.299554135013219,
        'test_agg_mean': 377.9968064884045
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20.0,
        'max_on_power': 3999.0,
        'train_agg_mean': 517.5859340919116,
        'train_agg_std': 827.1565574135092,
        'train_app_mean': 22.22078550102201,
        'train_app_std': 189.70389890256996,
        'test_app_mean': 29.433812118685246,
        'test_agg_mean': 685.6151694157477
    }
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

            # Calculate window center index.
            self.window_center = int(0.5 * (window_length - 1))

            # Number of input samples adjusted for windowing.
            # This prevents partial window generation.
            self.num_samples = self.total_samples - window_length

            # Generate indices of adjusted input sample array.
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
                [self.X[row:row + self.window_length] for row in rows])

            # Reshape samples to match model's input tensor format.
            # Starting shape = (batch_size, window_length)
            # Desired shape = (batch_size, 1, window_length)
            #samples = samples[:, np.newaxis, :]

            if self.train:
                # Create batch of single point targets from center of window.
                targets = np.array(
                    [self.y[row + self.window_center] for row in rows])
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
"""
Convert trained Keras models to tflite.

Generates a quantized tflite model that can be complied for the edge tpu
or used as-is for on device inference on a Raspberry Pi and other edge compute.

Copyright (c) 2022 Lindo St. Angel.
"""

from functools import partial
import os
import argparse
import socket

import tensorflow as tf
import numpy as np

import nilm_metric as nm
import common
from logger import log

# Number of samples for post-training quantization calibration.
NUM_CAL = 1000
# Dataset energy sampling period in seconds. 
SAMPLE_PERIOD = 8
# Name of best model checkpoint directory during training.
CHECKPOINT_DIR = 'checkpoints'
# Name of best pruned checkpoint directory during training.
PRUNED_CHECKPOINT_DIR = 'pruned_model_for_export'

rng = np.random.default_rng()

def convert(model, provider, num_cal, io_float=False):
    """ Convert Keras model to tflite.
    
    Optimize for latency and size, quantize weights and activations
    to int8 and optionally quantize input and output layers.

    Returns a quantized tflite model that can be complied for the edge tpu.

    Args:
        model: Keras input model.
        provider: Object that provides samples for calibration.
        num_cal: Number of calibration passes.
        io_float: Boolean indicating float input and outputs.

    Returns:
        Quantized tflite model.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Converter optimization settings. 
    # The DEFAULT optimization strategy quantizes model weights and
    # tries to optimize for size and latency, while minimizing the
    # loss in accuracy.
    # The EXPERIMENTAL_SPARSITY optimization tries to advantage of
    # the sparse model weights trained with pruning if available.
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT,
        tf.lite.Optimize.EXPERIMENTAL_SPARSITY
    ]
    # ref_gen must be a callable so use partial to set parameters. 
    ref_gen = partial(
        representative_dataset_gen,
        provider=provider,
        num_cal=num_cal)
    converter.representative_dataset = tf.lite.RepresentativeDataset(ref_gen)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if not io_float:
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    return converter.convert()

def change_model_batch_size(model, batch_size=1):
    """Change a model's batch size."""
    model_config = model.get_config()

    # Get the layer config to modify.
    layer_0_config = model_config['layers'][0]

    # Change batch size.
    lst = list(layer_0_config['config']['batch_input_shape'])
    lst[0] = batch_size
    layer_0_config['config']['batch_input_shape'] = tuple(lst)

    # Apply changes to layers.
    model_config['layers'][0] = layer_0_config
    
    # Create new model based on new config.
    new_model = model.__class__.from_config(model_config, custom_objects={})
    # Apply weights from original model to new model.
    new_model.set_weights(model.get_weights())

    return new_model

def representative_dataset_gen(provider, num_cal) -> np.float32:
    """Outputs samples from dataset at random locations"""
    # Get number of samples in provider.
    provider_len = provider.__len__()
    # Calculate num_cal sized indices of random locations.
    indices = rng.choice(provider_len, size=num_cal)
    # Generate samples from provider.
    for i in indices:
        sample, _ = provider.__getitem__(i)
        # Add axis to match model InputLayer.
        sample = sample[:, :, :, np.newaxis]
        yield [sample]

def evaluate_tflite(args, appliance_name, tflite_model, provider):
    """Evaluate a converted tflite model"""
    # Load dataset.
    test_file_name = common.find_test_filename(
        args.datadir, appliance_name, args.test_type)
    dataset_path = os.path.join(args.datadir, appliance_name, test_file_name)
    log(f'dataset: {dataset_path}')
    dataset = common.load_dataset(dataset_path, args.crop)
    num_samples = dataset[0].size
    log(f'Loaded {num_samples/10**6:.3f}M samples from dataset.')

    # Start the tflite interpreter.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    # Perform inference.
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

    del dataset

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Convert Keras models to tflite.')
    parser.add_argument(
        '--appliance_name',
        type=str,
        default='kettle',
        help='Name of target appliance.')
    parser.add_argument(
        '--datadir',
        type=str,
        default='/home/lindo/Develop/nilm/ml/dataset_management/refit',
        help='Directory of datasets.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/home/lindo/Develop/nilm/ml/models',
        help='Directory to save the tflite model')
    parser.add_argument(
        '--crop',
        type=int,
        default=None,
        help='Number of dataset samples to use. Default uses entire dataset.')
    parser.add_argument(
        '--evaluate', action='store_true',
        help='If set, evaluate tflite model for accuracy.')
    parser.add_argument(
        '--num_eval',
        type=int,
        default=100000,
        help='Number of inferences used to evaluate quantized model.')
    parser.add_argument(
        '--test_type',
        type=str,
        default='test',
        help='Type of the test set to load: \
            test -- test on the proper test set;\
            train -- test on a already prepared slice of the train set;\
            val -- test on the validation set;\
            uk -- test on UK-DALE;\
            redd -- test on REDD.')
    parser.add_argument(
        '--io_float', action='store_true',
        help='If set, make tflite I/O float.')
    parser.add_argument(
        '--prune', action='store_true',
        help='If set, convert a pruned model to tflite.')
    parser.set_defaults(io_float=False)
    parser.set_defaults(evaluate=False)
    parser.set_defaults(prune=False)
    return parser.parse_args()

if __name__ == '__main__':
    log(f'Machine name: {socket.gethostname()}')
    args = get_arguments()
    log('Arguments: ')
    log(args)

    # The appliance to train on.
    appliance_name = args.appliance_name

    if args.prune:
        # Load Keras model from best pruned checkpoint during training.
        model_filepath = os.path.join(
            args.save_dir, appliance_name, PRUNED_CHECKPOINT_DIR)
    else:
        # Load Keras model from best checkpoint during training.
        model_filepath = os.path.join(
            args.save_dir, appliance_name, CHECKPOINT_DIR)

    log(f'Model file path: {model_filepath}')
    original_model = tf.keras.models.load_model(model_filepath)
    original_model.summary()

    # Change loaded model batch size from None to 1.
    # This will make the batch size static for use in tpu compilation.
    # The edge tpu compiler accepts only static batch sizes.
    model = change_model_batch_size(original_model)
    model.summary()

    # Load dataset.
    test_file_name = common.find_test_filename(
        args.datadir, appliance_name, args.test_type)
    dataset_path = os.path.join(
        args.datadir, appliance_name, test_file_name)
    log(f'dataset: {dataset_path}')
    dataset = common.load_dataset(dataset_path, args.crop)
    num_samples = dataset[0].size
    log(f'Loaded {num_samples/10**6:.3f}M samples from dataset.')

    # Provider of windowed dataset samples and single point targets.
    WindowGenerator = common.get_window_generator()
    provider = WindowGenerator(
        dataset=dataset,
        batch_size=1,
        shuffle=False)

    # Convert model to tflite and quantize.
    tflite_model_quant = convert(
        model=model,
        provider=provider,
        num_cal=NUM_CAL,
        io_float=args.io_float)

    # Save converted model.
    if args.io_float:
        s = f'{appliance_name}_quant_flt.tflite'
    else:
        s = f'{appliance_name}_quant.tflite'
    filepath = os.path.join(args.save_dir, appliance_name, s)
    with open(filepath, 'wb') as file:
        file.write(tflite_model_quant)
    log(f'Quantized tflite model saved to {filepath}.')

    if args.evaluate:
        evaluate_tflite(args, appliance_name, tflite_model_quant, provider)
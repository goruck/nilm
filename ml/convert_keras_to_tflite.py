"""
Convert trained Keras models to tflite.

Generates a quantized tflite model that can be complied for the edge tpu
or used as-is for on device inference on a Raspberry Pi and other edge compute.

Copyright (c) 2022~2023 Lindo St. Angel.
"""

from functools import partial
import os
import argparse
import socket

import tensorflow as tf
import numpy as np

import nilm_metric
import common
from logger import log

# Number of samples for post-training quantization calibration.
NUM_CAL = 10800 # 24 hours @ 8 sec per sample
# Number of samples from the start of cal dataset to start calibration.
CAL_OFFSET = 2 * 10800
# Number of threads used by the interpreter and available to CPU kernels.
NUM_INTERPRETER_THREADS = 8
# Name of best pruned checkpoint directory during training.
PRUNED_CHECKPOINT_DIR = 'pruned_model_for_export'

rng = np.random.default_rng()

def convert(model, provider, num_cal, cal_offset, io_float):
    """ Convert Keras model to tflite.
    
    Optimize for latency and size, quantize weights and activations
    to int8 and optionally quantize input and output layers.

    Returns a quantized tflite model that can be complied for the edge tpu.

    Args:
        model: Keras input model.
        provider: Object that provides samples for calibration.
        num_cal: Number of calibration passes.
        cal_offset: Sample number of data set to start calibration.
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
        num_cal=num_cal,
        cal_offset=cal_offset)
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

def representative_dataset_gen(provider, num_cal=10800, cal_offset=0) -> np.float32:
    """Yields samples from representative dataset.

    This is a generator function that provides a small dataset to calibrate or
    estimate the range, i.e, (min, max) of all floating-point arrays in the model.
    
    Since dataset is very imbalanced must ensure enough samples are used to get
    generate a representative set. 24 hours of samples is a good starting point.

    Args:
        provider: Object that provides samples for calibration.
        num_cal: Number of calibration passes.
        cal_offset: Sample number of data set to start calibration.

    Yields:
        A representative model input sample.
    """
    # Get number of samples per batch in provider. Since batch should always be
    # set to 1 for inference, this will simply return the total number of samples.
    samples_per_batch = provider.__len__()

    # Calculate num_eval sized indices of contiguous locations in provider,
    # starting from cal_offset.
    if num_cal - cal_offset > samples_per_batch:
        raise ValueError('Not enough representative samples.')
    indices = list(range(samples_per_batch))[cal_offset:num_cal+cal_offset]

    # Calculate num_cal sized indices of random locations in dataset.
    #indices = rng.choice(samples_per_batch, size=num_cal)

    # Generate samples from provider.
    for i in indices:
        sample, _, _ = provider.__getitem__(i)
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
    interpreter = tf.lite.Interpreter(
        model_content=tflite_model,
        num_threads=NUM_INTERPRETER_THREADS)
    
    # Perform inference.
    results = common.tflite_infer(
        interpreter=interpreter,
        provider=provider,
        num_eval=args.num_eval,
        log=log)

    ground_truth = np.array([g for g, _ in results])
    prediction = np.array([p for _, p in results])

    # De-normalize appliance power predictions.
    if common.USE_APPLIANCE_NORMALIZATION:
        app_mean = 0
        app_std = common.params_appliance[appliance_name]['max_on_power']
    else:
        train_app_mean = common.params_appliance[appliance_name]['train_app_mean']
        train_app_std = common.params_appliance[appliance_name]['train_app_std']
        alt_app_mean = common.params_appliance[appliance_name]['alt_app_mean']
        alt_app_std = common.params_appliance[appliance_name]['alt_app_std']
        app_mean = alt_app_mean if common.USE_ALT_STANDARDIZATION else train_app_mean
        app_std = alt_app_std if common.USE_ALT_STANDARDIZATION else train_app_std
        print('Using alt standardization.' if common.USE_ALT_STANDARDIZATION
              else 'Using default standardization.')
        
    log(f'De-normalizing predictions with mean = {app_mean} and std = {app_std}.')

    prediction = prediction * app_std + app_mean
    ground_truth = ground_truth * app_std + app_mean

    # Apply on-power threshold.
    appliance_threshold = common.params_appliance[appliance_name]['on_power_threshold']
    log(f'appliance threshold: {appliance_threshold}')
    prediction[prediction < appliance_threshold] = 0.0

    sample_period = common.SAMPLE_PERIOD

    # Calculate ground truth and prediction status.
    prediction_status = np.array(common.compute_status(prediction, appliance_name))
    ground_truth_status = np.array(common.compute_status(ground_truth, appliance_name))
    assert prediction_status.size == ground_truth_status.size

    # Metric evaluation.
    metrics = nilm_metric.NILMTestMetrics(target=ground_truth,
                                          target_status=ground_truth_status,
                                          prediction=prediction,
                                          prediction_status=prediction_status,
                                          sample_period=sample_period)
    log(f'True positives: {metrics.get_tp()}')
    log(f'True negatives: {metrics.get_tn()}')
    log(f'False positives: {metrics.get_fp()}')
    log(f'False negatives: {metrics.get_fn()}')
    log(f'Accuracy: {metrics.get_accuracy()}')
    log(f'MCC: {metrics.get_mcc()}')
    log(f'F1: {metrics.get_f1()}')
    log(f'MAE: {metrics.get_abs_error()["mean"]} (W)')
    log(f'NDE: {metrics.get_nde()}')
    log(f'SAE: {metrics.get_sae()}')
    log(f'Ground truth EPD: {nilm_metric.get_epd(ground_truth * ground_truth_status, sample_period)} (Wh)')
    log(f'Predicted EPD: {nilm_metric.get_epd(prediction * prediction_status, sample_period)} (Wh)')

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
        '--model_arch',
        type=str,
        default='cnn',
        help='network architecture to use')
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
        default=432000, # 960 hrs (40 days) @ 8 sec per sample
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
    parser.add_argument(
        '--use_tpu', action='store_true',
        help='If set, prepare model for edge tpu compilation.')
    parser.set_defaults(io_float=False)
    parser.set_defaults(evaluate=False)
    parser.set_defaults(prune=False)
    parser.set_defaults(use_tpu=False)
    return parser.parse_args()

if __name__ == '__main__':
    log(f'Machine name: {socket.gethostname()}')
    args = get_arguments()
    log('Arguments: ')
    log(args)

    # The appliance to train on.
    appliance_name = args.appliance_name

    model_filepath = os.path.join(args.save_dir, appliance_name)
    if args.prune:
        # Load Keras model from best pruned checkpoint during training.
        pruened_filepath = os.path.join(model_filepath, PRUNED_CHECKPOINT_DIR)
        model = tf.keras.models.load_model(pruened_filepath)
    else:
        # Load Keras model from best SaveModel during training.
        savemodel_filepath = os.path.join(model_filepath, f'savemodel_{args.model_arch}')
        log(f'Savemodel file path: {savemodel_filepath}')
        model = tf.keras.models.load_model(savemodel_filepath)

    # Prepare model for edge TPU compilation. Since the edge TPU complier
    # requires static batch sizes, change loaded model batch size from None to 1.
    # This is currently only supported for the cnn model architecture.
    if args.use_tpu:
        if args.model_arch == 'cnn':
            model = change_model_batch_size(model)
        else:
            raise ValueError('Edge TPU compilation not supported for {args.model_arch}')

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
        batch_size=1, # batch size must be 1 for inference
        shuffle=False)

    # Convert model to tflite and quantize.
    log('Converting model to tflite format and quantizing.')
    tflite_model_quant = convert(
        model=model,
        provider=provider,
        num_cal=NUM_CAL,
        cal_offset=CAL_OFFSET,
        io_float=args.io_float)

    # Save converted model.
    base_name = f'{appliance_name}_{args.model_arch}_quant'
    filepath = os.path.join(args.save_dir,
                            appliance_name,
                            f'{base_name}_flt.tflite' if args.io_float
                            else f'{base_name}.tflite')
    with open(filepath, 'wb') as file:
        file.write(tflite_model_quant)
    log(f'Quantized tflite model saved to {filepath}.')

    if args.evaluate:
        evaluate_tflite(args, appliance_name, tflite_model_quant, provider)
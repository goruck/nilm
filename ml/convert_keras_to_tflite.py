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
import matplotlib.pyplot as plt
import pandas as pd

import nilm_metric
import common
from logger import Logger

# Number of samples for post-training quantization calibration.
NUM_CAL = 4 * 10800 # 96 hours @ 8 sec per sample
# Number of threads used by the interpreter and available to CPU kernels.
NUM_INTERPRETER_THREADS = 8
# Name of best pruned checkpoint directory during training.
PRUNED_CHECKPOINT_DIR = 'pruned_model_for_export'
# Check how well model was quantized.
DEBUG = True
# Improve model accuracy at expense of performance (DEBUG must be True).
FIX_MODEL = False
# RMSE / scale quantization metric threshold.
HIGH_QUANT_ERROR = 0.7

rng = np.random.default_rng()

def convert(
        model,
        provider,
        num_cal,
        io_float,
        use_tpu,
        debug,
        debug_results_file,
        fix_model):
    """ Convert Keras model to tflite.
    
    Optimize for latency and size, quantize weights and activations
    to int8 and optionally quantize input and output layers.

    Returns a quantized tflite model that can be complied for the edge tpu.

    Args:
        model: Keras input model.
        provider: Object that provides samples for calibration.
        num_cal: Number of calibration passes.
        io_float: Boolean indicating float input and outputs.
        use_tpu: Make conversion compatible with edge tpu.
        debug: Debug model, comparing float with quantized.
        fix_mode: Attempt to improve model accuracy at expense of latency.

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
        #tf.lite.Optimize.EXPERIMENTAL_SPARSITY # will not work with debugger
    ]
    # rep_gen must be a callable so use partial to set parameters. 
    rep_gen = partial(
        representative_dataset_gen,
        provider=provider,
        num_cal=num_cal)
    converter.representative_dataset = tf.lite.RepresentativeDataset(rep_gen)
    if use_tpu:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if not io_float:
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    quantized_model = converter.convert()

    if debug:
        logger.log('Debugging model..')
        debugger = tf.lite.experimental.QuantizationDebugger(
            converter=converter,
            debug_dataset=rep_gen)
        
        debugger.run()

        with open(debug_results_file, 'w') as f:
            debugger.layer_statistics_dump(f)

        layer_stats = pd.read_csv(debug_results_file)
        logger.log(layer_stats)

        layer_stats['range'] = 255.0 * layer_stats['scale']
        layer_stats['rmse/scale'] = layer_stats.apply(
            lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)
        logger.log(layer_stats[['op_name', 'range', 'rmse/scale']])

        plt.figure(figsize=(15, 5))
        ax1 = plt.subplot(121)
        ax1.bar(np.arange(len(layer_stats)), layer_stats['range'])
        ax1.set_ylabel('range')
        ax2 = plt.subplot(122)
        ax2.bar(np.arange(len(layer_stats)), layer_stats['rmse/scale'])
        ax2.set_ylabel('rmse/scale')
        plt.show()

        # The RMSE / scale is close to 1 / sqrt(12) (~ 0.289) when quantized
        # distribution is similar to the original float distribution, indicating
        # a well quantized model. The larger the value is, it's more likely for
        # the layer not being quantized well. These layers can remain in float to
        # generate a selectively quantized model that increases accuracy at the 
        # expense of inference performance. See:
        #  https://www.tensorflow.org/lite/performance/quantization_debugger
        suspected_layers = list(
            layer_stats[layer_stats['rmse/scale'] > HIGH_QUANT_ERROR]['tensor_name'])
        logger.log(f'Suspected layers: {suspected_layers}')

        if fix_model:
            logger.log('Restoring suspected quantized layers to float.')
            debug_options = tf.lite.experimental.QuantizationDebugOptions(
                denylisted_nodes=suspected_layers)
            debugger = tf.lite.experimental.QuantizationDebugger(
                converter=converter,
                debug_dataset=rep_gen,
                debug_options=debug_options)
            
            selective_quantized_model = debugger.get_nondebug_quantized_model()

            return selective_quantized_model
    
    return quantized_model

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

def representative_dataset_gen(provider, num_cal=43200) -> np.float32:
    """Yields samples from representative dataset.

    This is a generator function that provides a small dataset to calibrate or
    estimate the range, i.e, (min, max) of all floating-point arrays in the model.
    
    Since dataset is very imbalanced must ensure enough samples are used to generate a
    representative set. 96 hours (4 days) of samples is a good starting point as that
    period should be long enough to capture infrequently used appliances.

    Args:
        provider: Object that provides samples for calibration.
        num_cal: Number of calibration samples.

    Yields:
        A representative model input sample.
    """
    # Get number of samples per batch in provider. Since batch should always be
    # set to 1 for inference, this will simply return the total number of samples.
    samples_per_batch = provider.__len__()

    if num_cal > samples_per_batch:
        raise ValueError('Not enough representative samples.')

    """
    # Generate 'num_cal' samples at random locations in dataset. 
    indices = rng.choice(samples_per_batch, size=num_cal)
    for i in indices:
        sample, _, _ = provider.__getitem__(i)
        yield [sample,]
    """
    
    # Generate 'num_cal' samples from provider.
    # Given the dataset is unbalanced, this logic ensures that sufficient
    # representative samples from active appliances periods are yielded.
    active_frac = 0.5 # fraction of cal samples during appliance active times
    num_cal_active = int(active_frac * num_cal)
    i = 0
    active = 0
    inactive = 0
    while True:
        if i > samples_per_batch: # out of samples
            break
        sample, _, status = provider.__getitem__(i)
        if sample.size == 0 or status.size == 0:
            raise RuntimeError(f'Missing representative data at i={i}.',)
        i+=1
        status = bool(status.item())
        if active < num_cal_active:
            if status:
                active+=1
                yield [sample,] # generate samples from active times...
            continue
        elif active + inactive < num_cal:
            if not status:
                inactive+=1
                yield [sample,] # ... then generate samples from inactive times
        else:
            break

def evaluate_tflite(args, appliance_name, tflite_model, provider):
    """Evaluate a converted tflite model"""
    # Load dataset.
    test_file_name = common.find_test_filename(
        args.datadir, appliance_name, args.test_type)
    dataset_path = os.path.join(args.datadir, appliance_name, test_file_name)
    logger.log(f'dataset: {dataset_path}')
    dataset = common.load_dataset(dataset_path, args.crop)
    num_samples = dataset[0].size
    logger.log(f'Loaded {num_samples/10**6:.3f}M samples from dataset.')

    # Start the tflite interpreter.
    interpreter = tf.lite.Interpreter(
        model_content=tflite_model,
        num_threads=NUM_INTERPRETER_THREADS)
    
    # Perform inference.
    results = common.tflite_infer(
        interpreter=interpreter,
        provider=provider,
        num_eval=args.num_eval,
        log=logger.log)

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
        logger.log('Using alt standardization.' if common.USE_ALT_STANDARDIZATION 
                   else 'Using default standardization.')
        
    logger.log(f'De-normalizing predictions with mean = {app_mean} and std = {app_std}.')

    prediction = prediction * app_std + app_mean
    ground_truth = ground_truth * app_std + app_mean

    # Apply on-power threshold.
    appliance_threshold = common.params_appliance[appliance_name]['on_power_threshold']
    logger.log(f'appliance threshold: {appliance_threshold}')
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
    logger.log(f'True positives: {metrics.get_tp()}')
    logger.log(f'True negatives: {metrics.get_tn()}')
    logger.log(f'False positives: {metrics.get_fp()}')
    logger.log(f'False negatives: {metrics.get_fn()}')
    logger.log(f'Accuracy: {metrics.get_accuracy()}')
    logger.log(f'MCC: {metrics.get_mcc()}')
    logger.log(f'F1: {metrics.get_f1()}')
    logger.log(f'MAE: {metrics.get_abs_error()["mean"]} (W)')
    logger.log(f'NDE: {metrics.get_nde()}')
    logger.log(f'SAE: {metrics.get_sae()}')
    epd_gt = nilm_metric.get_epd(ground_truth * ground_truth_status, sample_period)
    logger.log(f'Ground truth EPD: {epd_gt} (Wh)')
    epd_pred = nilm_metric.get_epd(prediction * prediction_status, sample_period)
    logger.log(f'Predicted EPD: {epd_pred} (Wh)')
    logger.log(f'EPD Relative Error: {100.0 * (epd_pred - epd_gt) / epd_gt} (%)')

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
    args = get_arguments()
    appliance_name = args.appliance_name
    logger = Logger(os.path.join(args.save_dir,
                                 appliance_name,
                                 f'{appliance_name}_quant_{args.model_arch}.log'))
    logger.log(f'Machine name: {socket.gethostname()}')
    logger.log('Arguments: ')
    logger.log(args)

    model_filepath = os.path.join(args.save_dir, appliance_name)
    if args.prune:
        # Load Keras model from best pruned checkpoint during training.
        pruened_filepath = os.path.join(model_filepath, PRUNED_CHECKPOINT_DIR)
        model = tf.keras.models.load_model(pruened_filepath)
    else:
        # Load Keras model from best SaveModel during training.
        savemodel_filepath = os.path.join(model_filepath, f'savemodel_{args.model_arch}')
        logger.log(f'Savemodel file path: {savemodel_filepath}')
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
    logger.log(f'dataset: {dataset_path}')
    dataset = common.load_dataset(dataset_path, args.crop)
    num_samples = dataset[0].size
    logger.log(f'Loaded {num_samples/10**6:.3f}M samples from dataset.')

    # Provider of windowed dataset samples and single point targets.
    WindowGenerator = common.get_window_generator()
    provider = WindowGenerator(
        dataset=dataset,
        batch_size=1, # batch size must be 1 for inference
        shuffle=False)

    # Convert model to tflite and quantize.
    logger.log('Converting model to tflite format and quantizing.')
    tflite_model_quant = convert(
        model=model,
        provider=provider,
        num_cal=NUM_CAL,
        io_float=args.io_float,
        use_tpu=args.use_tpu,
        debug=DEBUG,
        debug_results_file=os.path.join(
            args.save_dir, appliance_name, 'debug_results.csv'),
        fix_model=FIX_MODEL)

    # Save converted model.
    base_name = f'{appliance_name}_{args.model_arch}_quant'
    filepath = os.path.join(args.save_dir,
                            appliance_name,
                            f'{base_name}_flt.tflite' if args.io_float
                            else f'{base_name}.tflite')
    with open(filepath, 'wb') as file:
        file.write(tflite_model_quant)
    logger.log(f'Quantized tflite model saved to {filepath}.')

    if args.evaluate:
        evaluate_tflite(args, appliance_name, tflite_model_quant, provider)
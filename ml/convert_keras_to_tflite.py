"""
Convert trained Keras models to tflite.

Generates a quantized tflite model that can be complied for the edge tpu
or used for on-device inference on a Raspberry Pi or other edge compute.

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

rng = np.random.default_rng()

def convert(
        keras_model,
        sample_provider,
        quant_mode,
        debug_results_path,
        log):
    """ Convert Keras model to tflite.

    Optimize for latency and size, quantize weights and activations
    and optionally quantize input and output layers.

    Returns a quantized tflite model.

    Args:
        keras_model: Keras input model.
        sample_provider: Object that provides samples for calibration.
        quant_mode: Quantization mode.
        debug_results_path: Save path to debug results.
        log: Logger object.

    Returns:
        Quantized tflite model.
    """

    # Number of samples for post-training quantization calibration.
    num_cal = 4 * 10800 # 4 * 24 = 96 hours @ 8 sec per sample
    # Check how well model was quantized (only for full INT8 quant).
    debug = False
    # Improve model accuracy at expense of performance (debug must be True).
    fix_model = False

    if debug and quant_mode in ['w8', 'w8_a16']:
        raise ValueError('Model debug only works for full INT8 quant.')

    if fix_model and not debug:
        raise ValueError('Can only fix model if debug is set.')

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    # Converter optimization settings.
    # The DEFAULT optimization strategy quantizes model weights and
    # activations and tries to optimize for size and latency, while
    # minimizing the loss in accuracy.
    # The EXPERIMENTAL_SPARSITY optimization tries to advantage of
    # the sparse model weights trained with pruning if available.
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT,
        #tf.lite.Optimize.EXPERIMENTAL_SPARSITY # will not work with debugger
    ]

    rep_gen = partial(
        representative_dataset_gen,
        sample_provider=sample_provider,
        num_cal=num_cal,
        log=log
    )

    if quant_mode == 'w8':
        # Quantize only weights from floating point to int8.
        # Inputs and outputs are kept in float.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    elif quant_mode == 'w8_a8_fallback':
        # Quantize weights and activations from floating point to int8 with
        # fallback to float if an operator does not have an int implementation.
        # Inputs and outputs are kept in float.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.representative_dataset = tf.lite.RepresentativeDataset(rep_gen)
    elif quant_mode == 'w8_a8':
        # Enforce full int8 quantization for all ops including the input and output.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = tf.lite.RepresentativeDataset(rep_gen)
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif quant_mode == 'w8_a16':
        # Quantize activations based on their range to int16, weights
        # to int8 and bias to int64 with fallback to float for
        # unsupported operators. Inputs and outputs are kept in float.
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
        converter.representative_dataset = tf.lite.RepresentativeDataset(rep_gen)
    else:
        raise ValueError('Unrecognized quantization mode.')

    quantized_model = converter.convert()

    if debug:
        log.log('Debugging model...')
        debugger = tf.lite.experimental.QuantizationDebugger(
            converter=converter,
            debug_dataset=rep_gen)

        debugger.run()

        debug_results_file = os.path.join(debug_results_path, 'debug_results.csv')

        with open(debug_results_file, mode='w', encoding='utf-8') as f:
            debugger.layer_statistics_dump(f)

        pd.set_option('display.max_rows', None)
        layer_stats = pd.read_csv(debug_results_file)
        log.log(layer_stats)

        layer_stats['range'] = 255.0 * layer_stats['scale']
        layer_stats['rmse/scale'] = layer_stats.apply(
            lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)
        log.log(layer_stats[['op_name', 'range', 'rmse/scale']])

        plt.figure(figsize=(15, 5))
        ax1 = plt.subplot(121)
        ax1.bar(np.arange(len(layer_stats)), layer_stats['range'])
        ax1.set_ylabel('range')
        ax2 = plt.subplot(122)
        ax2.bar(np.arange(len(layer_stats)), layer_stats['rmse/scale'])
        ax2.set_ylabel('rmse/scale')
        debug_results_plot = os.path.join(debug_results_path, 'debug_results_plot.png')
        log.log(f'Saving debug results plot to {debug_results_plot}.')
        plt.savefig(fname=debug_results_plot)

        # The RMSE / scale is close to 1 / sqrt(12) (~ 0.289) when quantized
        # distribution is similar to the original float distribution, indicating
        # a well quantized model. The larger the value is, it's more likely for
        # the layer not being quantized well. These layers can remain in float to
        # generate a selectively quantized model that increases accuracy at the
        # expense of inference performance. See:
        #  https://www.tensorflow.org/lite/performance/quantization_debugger
        high_quant_error = 0.7 # RMSE / scale quantization metric threshold.
        suspected_layers = list(
            layer_stats[layer_stats['rmse/scale'] > high_quant_error]['tensor_name'])
        # Keep layers with range greater than 255 (8-bits) in float as well.
        #suspected_layers.extend(
            #list(layer_stats[layer_stats['range'] > 255.0]['tensor_name']))
        # Let the first 5 layers remain in float as well.
        #suspected_layers.extend(list(layer_stats[:5]['tensor_name']))
        log.log(f'Suspected layers: {suspected_layers}')

        if fix_model:
            log.log('Restoring suspected quantized layers to float.')
            debug_options = tf.lite.experimental.QuantizationDebugOptions(
                denylisted_nodes=suspected_layers)
                #denylisted_ops=['MUL','MEAN','RSQRT','SUB','ADD'])
            debugger = tf.lite.experimental.QuantizationDebugger(
                converter=converter,
                debug_dataset=rep_gen,
                debug_options=debug_options)

            selective_quantized_model = debugger.get_nondebug_quantized_model()

            return selective_quantized_model

    return quantized_model

def change_model_batch_size(input_model, batch_size=1):
    """Change a model's batch size."""

    model_config = input_model.get_config()

    # Get the layer config to modify.
    layer_0_config = model_config['layers'][0]

    # Change batch size.
    lst = list(layer_0_config['config']['batch_input_shape'])
    lst[0] = batch_size
    layer_0_config['config']['batch_input_shape'] = tuple(lst)

    # Apply changes to layers.
    model_config['layers'][0] = layer_0_config

    # Create new model based on new config.
    new_model = input_model.__class__.from_config(model_config, custom_objects={})
    # Apply weights from original model to new model.
    new_model.set_weights(input_model.get_weights())

    return new_model

def representative_dataset_gen(sample_provider, num_cal, log) -> np.float32:
    """Yields samples from representative dataset.

    This is a generator function that provides a small dataset to calibrate or
    estimate the range, i.e, (min, max) of all floating-point arrays in the model.

    Since dataset is very imbalanced must ensure enough samples are used to generate a
    representative set. 96 hours (4 days) of samples is a good starting point as that
    period should be long enough to capture infrequently used appliances.

    Args:
        sample_provider: Object that provides samples for calibration.
        num_cal: Number of calibration samples.
        log: Logger object.

    Yields:
        A representative model input sample.
    """

    # Get number of samples per batch in provider. Since batch should always be
    # set to 1 for inference, this will simply return the total number of samples.
    samples_per_batch = sample_provider.__len__()

    if num_cal > samples_per_batch:
        raise ValueError('Not enough representative samples.')

    # Generate 'num_cal' samples at random locations in dataset.
    #indices = rng.choice(samples_per_batch, size=num_cal)
    #for i in indices:
        #sample, _, _ = sample_provider.__getitem__(i)
        #yield [sample,]

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
            log.log(f'Out of cal samples, act={active}, inact={inactive}.',
                       level='warning')
            break

        sample, _, status = sample_provider.__getitem__(i)

        if sample.size == 0 or status.size == 0:
            raise RuntimeError(f'Missing representative data at i={i}.',)

        status = bool(status.item())
        i+=1

        if active < num_cal_active:
            if status:
                active+=1
                yield [sample,] # generate samples from active times...
            continue

        if active + inactive < num_cal:
            if not status:
                inactive+=1
                yield [sample,] # ... then generate samples from inactive times
        else:
            break

def evaluate_tflite(num_eval, appliance, tflite_model, sample_provider, log):
    """Evaluate a converted tflite model"""

    # Start the tflite interpreter.
    interpreter = tf.lite.Interpreter(
        model_content=tflite_model,
        num_threads=8 # CPU threads used by the interpreter
    )

    # Perform inference.
    results = common.tflite_infer(
        interpreter=interpreter,
        provider=sample_provider,
        num_eval=num_eval,
        log=log.log)

    ground_truth = np.array([g for g, _ in results])
    prediction = np.array([p for _, p in results])

    # De-normalize appliance power predictions.
    if common.USE_APPLIANCE_NORMALIZATION:
        app_mean = 0
        app_std = common.params_appliance[appliance]['max_on_power']
    else:
        train_app_mean = common.params_appliance[appliance]['train_app_mean']
        train_app_std = common.params_appliance[appliance]['train_app_std']
        alt_app_mean = common.params_appliance[appliance]['alt_app_mean']
        alt_app_std = common.params_appliance[appliance]['alt_app_std']
        app_mean = alt_app_mean if common.USE_ALT_STANDARDIZATION else train_app_mean
        app_std = alt_app_std if common.USE_ALT_STANDARDIZATION else train_app_std
        log.log(
            'Using alt standardization.' if common.USE_ALT_STANDARDIZATION
            else 'Using default standardization.'
        )

    log.log(f'De-normalizing predictions with mean = {app_mean} and std = {app_std}.')

    prediction = prediction * app_std + app_mean
    ground_truth = ground_truth * app_std + app_mean

    # Apply on-power threshold.
    appliance_threshold = common.params_appliance[appliance]['on_power_threshold']
    log.log(f'appliance threshold: {appliance_threshold}')
    prediction[prediction < appliance_threshold] = 0.0

    sample_period = common.SAMPLE_PERIOD

    # Calculate ground truth and prediction status.
    prediction_status = np.array(common.compute_status(prediction, appliance))
    ground_truth_status = np.array(common.compute_status(ground_truth, appliance))
    assert prediction_status.size == ground_truth_status.size

    # Metric evaluation.
    metrics = nilm_metric.NILMTestMetrics(
        target=ground_truth,
        target_status=ground_truth_status,
        prediction=prediction,
        prediction_status=prediction_status,
        sample_period=sample_period
    )
    log.log(f'True positives: {metrics.get_tp()}')
    log.log(f'True negatives: {metrics.get_tn()}')
    log.log(f'False positives: {metrics.get_fp()}')
    log.log(f'False negatives: {metrics.get_fn()}')
    log.log(f'Accuracy: {metrics.get_accuracy()}')
    log.log(f'MCC: {metrics.get_mcc()}')
    log.log(f'F1: {metrics.get_f1()}')
    log.log(f'MAE: {metrics.get_abs_error()["mean"]} (W)')
    log.log(f'NDE: {metrics.get_nde()}')
    log.log(f'SAE: {metrics.get_sae()}')
    epd_gt = nilm_metric.get_epd(ground_truth * ground_truth_status, sample_period)
    log.log(f'Ground truth EPD: {epd_gt} (Wh)')
    epd_pred = nilm_metric.get_epd(prediction * prediction_status, sample_period)
    log.log(f'Predicted EPD: {epd_pred} (Wh)')
    log.log(f'EPD Relative Error: {100.0 * (epd_pred - epd_gt) / epd_gt} (%)')

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Convert Keras models to tflite.'
    )
    parser.add_argument(
        '--appliance_name',
        type=str,
        default='kettle',
        choices=['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine'],
        help='Name of target appliance.'
    )
    parser.add_argument(
        '--model_arch',
        type=str,
        default='cnn',
        choices=['cnn', 'transformer'],
        help='Network architecture to use'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='/home/lindo/Develop/nilm/ml/dataset_management/refit',
        help='Directory of datasets'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/home/lindo/Develop/nilm/ml/models',
        help='Directory to save the tflite model'
    )
    parser.add_argument(
        '--crop',
        type=int,
        default=None,
        help='Number of dataset samples to use. Default uses entire dataset'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='If set, evaluate tflite model for accuracy'
    )
    parser.add_argument(
        '--num_eval',
        type=int,
        default=432000, # 960 hrs (40 days) @ 8 sec per sample
        help='Number of inferences used to evaluate quantized model'
    )
    parser.add_argument(
        '--test_type',
        type=str,
        default='test',
        choices=['test', 'train', 'val', 'uk', 'redd'],
        help=(
            'Type of the test set to load: '
            'test - test on the proper test set '
            'train - test on a already prepared slice of the train set '
            'val - test on the validation set '
            'uk - test on UK-DALE '
            'redd - test on REDD'
        )
    )
    parser.add_argument(
        '--prune',
        action='store_true',
        help='If set, convert a pruned model to tflite.'
    )
    parser.add_argument(
        '--quant_mode',
        type=str,
        default='w8',
        choices=['w8', 'w8_a8_fallback', 'w8_a8', 'w8_a16'],
        help=(
            'Quantization mode: '
            'w8 - quantize weights only to int8 '
            'w8_a8_fallback - quantize weights and activations to int8 with fallback to float '
            'w8_a8 - quantize weights and activations to int8 '
            'w8_a16 - quantize weights to int8 and activations to int16'
        )
    )
    parser.set_defaults(evaluate=False)
    parser.set_defaults(prune=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    appliance_name = args.appliance_name
    logger = Logger(os.path.join(
        args.save_dir,appliance_name,
        f'{appliance_name}_{args.model_arch}_{args.quant_mode}.log')
    )
    logger.log(f'Machine name: {socket.gethostname()}')
    logger.log('Arguments: ')
    logger.log(args)

    model_filepath = os.path.join(args.save_dir, appliance_name)
    if args.prune:
        # Load Keras model from best pruned checkpoint during training.
        PRUNED_CHECKPOINT_DIR = 'pruned_model_for_export'
        pruned_filepath = os.path.join(model_filepath, PRUNED_CHECKPOINT_DIR)
        model = tf.keras.models.load_model(pruned_filepath)
    else:
        # Load Keras model from best SaveModel during training.
        savemodel_filepath = os.path.join(model_filepath, f'savemodel_{args.model_arch}')
        logger.log(f'Savemodel file path: {savemodel_filepath}')
        model = tf.keras.models.load_model(savemodel_filepath)

    # Prepare model for edge TPU compilation using w8_a8 quantization.
    # Since the edge TPU complier requires static batch sizes, change
    # loaded model batch size from None to 1.
    # This is currently only supported for the cnn model architecture.
    if args.quant_mode == 'w8_a8':
        if args.model_arch == 'cnn':
            model = change_model_batch_size(model)
        else:
            raise ValueError('w8_a8 quant not supported for {args.model_arch}')

    model.summary()

    # Load dataset.
    test_file_name = common.find_test_filename(
        args.datadir, appliance_name, args.test_type)
    dataset_path = os.path.join(
        args.datadir, appliance_name, test_file_name)
    logger.log(f'dataset: {dataset_path}')
    dataset = common.load_dataset(dataset_path, args.crop)
    logger.log(f'Loaded {dataset[0].size/10**6:.3f}M samples from dataset.')

    # Provider of windowed dataset samples and single point targets.
    WindowGenerator = common.get_window_generator()
    provider = WindowGenerator(
        dataset=dataset,
        batch_size=1, # batch size must be 1 for inference
        shuffle=False
    )

    # Convert model to tflite and quantize.
    logger.log(f'Converting model to tflite using {args.quant_mode} quantization.')
    tflite_model_quant = convert(
        keras_model=model,
        sample_provider=provider,
        quant_mode=args.quant_mode,
        debug_results_path=os.path.join(args.save_dir, appliance_name),
        log=logger
    )

    # Save converted model.
    filepath = os.path.join(
        args.save_dir,
        appliance_name,
        f'{appliance_name}_{args.model_arch}_{args.quant_mode}.tflite'
    )
    with open(filepath, 'wb') as file:
        file.write(tflite_model_quant)
    logger.log(f'Quantized tflite model saved to {filepath}.')

    if args.evaluate:
        evaluate_tflite(
            args.num_eval,
            appliance_name,
            tflite_model_quant,
            provider,
            logger
        )

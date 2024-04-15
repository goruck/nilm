"""
Convert Keras models trained for energy disaggregation to tflite.

Generates a quantized tflite model that can be complied for the edge tpu
or used for on-device inference on a Raspberry Pi or other edge compute.

Copyright (c) 2022~2024 Lindo St. Angel.
"""

import os
import argparse
import socket

import tensorflow as tf
import numpy as np

import common
from nilm_metric import NILMTestMetrics
from logger import Logger
from convert_model import ConvertModel
from window_generator import WindowGenerator

# Number of samples for post-training quantization calibration.
# 43200 (4 * 10800) is 4 * 24 = 96 hours @ 8 sec per sample.
NUM_CAL = 43200

# Name of purned model checkpoints.
PRUNED_CHECKPOINT_DIR = 'pruned_model_for_export'

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
    metrics = NILMTestMetrics(
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
    epd_gt = metrics.get_epd(ground_truth * ground_truth_status, sample_period)
    log.log(f'Ground truth EPD: {epd_gt} (Wh)')
    epd_pred = metrics.get_epd(prediction * prediction_status, sample_period)
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
        choices=['cnn', 'cnn_fine_tune', 'transformer'],
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
        default='convert_only',
        choices=['convert_only', 'w8', 'w8_a8_fallback', 'w8_a8', 'w8_a16'],
        help=(
            'Quantization mode: '
            'convert_only - no quantization '
            'w8 - quantize weights only to int8 '
            'w8_a8_fallback - quantize weights and activations to int8 with fallback to float '
            'w8_a8 - quantize weights and activations to int8 '
            'w8_a16 - quantize weights to int8 and activations to int16'
        )
    )
    parser.add_argument(
        '--debug_model',
        action='store_true',
        help='If set, check how well model was quantized (only for full INT8 quant).'
    )
    parser.add_argument(
        '--fix_model',
        action='store_true',
        help='If set, attempt to improve model accuracy at expense of performance.'
    )
    parser.add_argument(
        '--use_tpu',
        action='store_true',
        help='If set, make model compatible with edge TPU compilation.'
    )
    parser.set_defaults(evaluate=False)
    parser.set_defaults(prune=False)
    parser.set_defaults(debug_model=False)
    parser.set_defaults(fix_model=False)
    parser.set_defaults(use_tpu=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    appliance_name = args.appliance_name
    log_path = os.path.join(
        args.save_dir,
        appliance_name,
        f'{appliance_name}_{args.model_arch}_convert_{args.quant_mode}_fixed.log' if args.fix_model
        else f'{appliance_name}_{args.model_arch}_convert_{args.quant_mode}.log'
    )
    logger = Logger(log_file_name=log_path)
    logger.log(f'Machine name: {socket.gethostname()}')
    logger.log('Arguments: ')
    logger.log(args)

    model_filepath = os.path.join(args.save_dir, appliance_name)
    if args.prune:
        # Load Keras model from best pruned checkpoint during training.
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
    if args.use_tpu:
        if args.quant_mode == 'w8_a8' and args.model_arch == 'cnn':
            model = change_model_batch_size(model)
        else:
            raise ValueError('tpu config must use quant_mode `w8_a8` and model_arch `cnn`')

    # Check for currently unsupported conversions.
    # TODO: fix
    # INT16 conversions of transformer model currently lead to the following tflite interpreter runtime error:
    # `RuntimeError: tensorflow/lite/kernels/elementwise.cc:105 Type INT16 is unsupported by op Rsqrt.Node number 31 (RSQRT) failed to prepare.``
    if args.model_arch == 'transformer' and args.quant_mode == 'w8_a16':
        raise ValueError('transformer conversions with INT16 types are not supported')

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
    provider = WindowGenerator(
        dataset=dataset,
        batch_size=1, # batch size must be 1 for inference
        shuffle=False
    )

    # Convert keras model to tflite and quantize.
    logger.log(f'Converting model to tflite using {args.quant_mode} quantization.')
    convert_model = ConvertModel(
        keras_model=model,
        quant_mode=args.quant_mode,
        sample_provider=provider,
        num_cal=NUM_CAL,
        log=logger,
        debug_results_filepath=os.path.join(args.save_dir, appliance_name),
        debug_results_filename=f'{appliance_name}_{args.model_arch}_debug_{args.quant_mode}.csv',
        debug_results_plot_name=f'{appliance_name}_{args.model_arch}_debug_{args.quant_mode}.png',
    )
    if args.debug_model:
        # Convert model and check to see how well it was quantized.
        tflite_model_quant = convert_model.debug()
    elif args.fix_model:
        # Convert model and attempt to "fix" troublesome converted layers by
        # keeping them in float32.
        tflite_model_quant = convert_model.fix()
    else:
        # Just convert model.
        tflite_model_quant = convert_model.convert(
            set_input_type_int8 = args.use_tpu,
            set_output_type_int8 = args.use_tpu
        )

    # Save converted model.
    filepath = os.path.join(
        args.save_dir,
        appliance_name,
        f'{appliance_name}_{args.model_arch}_{args.quant_mode}_fixed.tflite' if args.fix_model
        else f'{appliance_name}_{args.model_arch}_{args.quant_mode}.tflite'
    )
    with open(filepath, 'wb') as file:
        file.write(tflite_model_quant)
    logger.log(f'Quantized tflite model saved to {filepath}.')

    # Evaluate quantized model performance.
    if args.evaluate:
        evaluate_tflite(
            args.num_eval,
            appliance_name,
            tflite_model_quant,
            provider,
            logger
        )

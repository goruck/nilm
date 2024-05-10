"""
Evaluate tflite model performance on raspberry pi.

Copyright (c) 2022 Lindo St. Angel.
"""

import os
import argparse
import socket

import tflite_runtime.interpreter as tflite
import numpy as np

import common
from nilm_metric import NILMTestMetrics
from logger import Logger
from window_generator import WindowGenerator

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Evaluate a tflite model on a raspberry pi.'
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
        choices=['cnn', 'transformer', 'fcn', 'resnet'],
        help='Network architecture to use.'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='../ml/dataset_management/refit',
        help='Directory of the test samples.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='../ml/models',
        help='Directory to save test results.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='../ml/models/',
        help='tflite model path.'
    )
    parser.add_argument(
        '--crop',
        type=int,
        default=None,
        help='Number of dataset samples to use. Default uses entire dataset.'
    )
    parser.add_argument(
        '--num_eval',
        type=int,
        default=100000,
        help='Number of inferences used to evaluate quantized model.'
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
        '--use_fixed_model',
        action='store_true',
        help='If set, use model fixed with selective qunatization.'
    )
    parser.set_defaults(use_fixed_model=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    appliance_name = args.appliance_name

    logger = Logger(
        log_file_name=os.path.join(
            args.save_dir,
            appliance_name,
            f'{appliance_name}_{args.model_arch}_{args.quant_mode}_rpi_eval.log'
        )
    )

    logger.log(f'Machine name: {socket.gethostname()}')
    logger.log('Arguments: ')
    logger.log(args)

    # Load tflite model.
    model_filepath = os.path.join(
        args.model_path,
        appliance_name,
        f'{appliance_name}_{args.model_arch}_{args.quant_mode}_fixed.tflite' if args.use_fixed_model
        else f'{appliance_name}_{args.model_arch}_{args.quant_mode}.tflite'
    )
    logger.log(f'tflite model: {model_filepath}')

    # Load dataset.
    test_file_name = common.find_test_filename(
        args.datadir, appliance_name, args.test_type
    )
    dataset_path = os.path.join(args.datadir, appliance_name, test_file_name)
    logger.log(f'dataset: {dataset_path}')
    dataset = common.load_dataset(dataset_path, args.crop)
    logger.log(f'Loaded {dataset[0].size/10**6:.3f}M samples from dataset.')

    # Provider of windowed dataset samples, targets and status.
    provider = WindowGenerator(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )

    # Perform inference on windowed samples.
    interpreter = tflite.Interpreter(model_path=model_filepath, num_threads=4)
    results = common.tflite_infer(
        interpreter=interpreter,
        provider=provider,
        num_eval=args.num_eval,
        log=logger.log
    )

    ground_truth = np.array([g for g, _ in results])
    prediction = np.array([p for _, p in results])

    # De-normalize appliance power predictions.
    train_app_std = common.params_appliance[appliance_name]['train_app_std']
    train_app_mean = common.params_appliance[appliance_name]['train_app_mean']
    logger.log(f'Train appliance mean: {train_app_mean} (W)')
    logger.log(f'Train appliance std: {train_app_std} (W)')
    if common.USE_APPLIANCE_NORMALIZATION:
        app_mean = 0
        app_std = common.params_appliance[appliance_name]['max_on_power']
    else:
        alt_app_mean = common.params_appliance[appliance_name]['alt_app_mean']
        alt_app_std = common.params_appliance[appliance_name]['alt_app_std']
        app_mean = alt_app_mean if common.USE_ALT_STANDARDIZATION else train_app_mean
        app_std = alt_app_std if common.USE_ALT_STANDARDIZATION else train_app_std
        logger.log(
            'Using alt standardization.' if common.USE_ALT_STANDARDIZATION
            else 'Using default standardization.'
        )
    logger.log(f'De-normalizing predictions with mean = {app_mean} and std = {app_std}.')
    prediction = prediction.flatten() * app_std + app_mean
    # Remove negative energy predictions
    prediction[prediction <= 0.0] = 0.0

    # De-normalize ground truth.
    ground_truth = ground_truth.flatten() * app_std + app_mean

    # Calculate ground truth and prediction status.
    prediction_status = np.array(common.compute_status(prediction, appliance_name))
    ground_truth_status = np.array(common.compute_status(ground_truth, appliance_name))

    metrics = NILMTestMetrics(
        target=ground_truth,
        target_status=ground_truth_status,
        prediction=prediction,
        prediction_status=prediction_status,
        sample_period=common.SAMPLE_PERIOD
    )
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
    epd_gt = metrics.get_epd(ground_truth * ground_truth_status, common.SAMPLE_PERIOD)
    logger.log(f'Ground truth EPD: {epd_gt} (Wh)')
    epd_pred = metrics.get_epd(prediction * prediction_status, common.SAMPLE_PERIOD)
    logger.log(f'Predicted EPD: {epd_pred} (Wh)')
    logger.log(f'EPD Relative Error: {100.0 * (epd_pred - epd_gt) / epd_gt} (%)')

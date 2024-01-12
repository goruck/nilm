"""
Read data from arduino and perform appliance power inference.

Copyright (c) 2022~2024 Lindo St. Angel.
"""

import os
from datetime import datetime
from time import time, sleep
from signal import signal, SIGINT
import csv
import socket
import sys
import argparse

import serial
import pytz
import tflite_runtime.interpreter as tflite
import numpy as np

sys.path.append('../ml')
from logger import Logger #pylint: disable=import-error
import common #pylint: disable=import-error

# Rpi serial port where the Arduino is connected.
SER_PORT = '/dev/ttyACM0'

# Rpi serial port baudrate.
SER_BAUDRATE = 115200

# These are the appliance types that the models will predict power for.
APPLIANCES = [
    'kettle', 'fridge', 'microwave', 'washingmachine', 'dishwasher'
]

# Row names of csv file that stores power readings and predictions.
CSV_ROW_NAMES = [
    'DT','TS', # datetime, unix timestamp(UTC)
    'V', # rms main voltage
    'I1','W1','VA1', # mains phase 1 rms current, real power, apparent power
    'I2','W2','VA2', # mains phase 2 rms current, real power, apparent power
] + APPLIANCES # predicted appliance powers

# Number of power mains samples to run inference on.
WINDOW_LENGTH = 599

# Model architecture and mode used to quantize it.
MODEL_ARCH = 'cnn'
QUANT_MODE = 'w8'

logger = Logger(level='info')

def get_scaling(app:str) -> tuple[float, float]:
    """Get appliance mean and std for normalization and de-normalization."""
    if common.USE_APPLIANCE_NORMALIZATION:
        return 0.0, common.params_appliance[app]['max_on_power']

    train_app_std = common.params_appliance[app]['train_app_std']
    train_app_mean = common.params_appliance[app]['train_app_mean']
    alt_app_mean = common.params_appliance[app]['alt_app_mean']
    alt_app_std = common.params_appliance[app]['alt_app_std']
    app_mean = alt_app_mean if common.USE_ALT_STANDARDIZATION else train_app_mean
    app_std = alt_app_std if common.USE_ALT_STANDARDIZATION else train_app_std
    return app_mean, app_std

def infer(app:str, data:np.ndarray, model_path:str) -> np.ndarray:
    """Perform inference using a tflite model."""
    logger.log(f'Predicting power for {app}', level='debug')

    # Start the tflite interpreter.
    try:
        interpreter = tflite.Interpreter(
            model_path=os.path.join(
                model_path, app, f'{app}_{MODEL_ARCH}_{QUANT_MODE}.tflite'
            )
        )
    except ValueError as e:
        logger.log(f'{app} tflite model not found', level='error')
        logger.log(e, level='error')
        return np.NaN

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    logger.log(f'interpreter input details: {input_details}', level='debug')
    output_details = interpreter.get_output_details()
    logger.log(f'interpreter output details: {output_details}', level='debug')

    # Check I/O tensor type.
    floating_input = input_details[0]['dtype'] == np.float32
    logger.log(f'tflite model floating input: {floating_input}', level='debug')
    floating_output = output_details[0]['dtype'] == np.float32
    logger.log(f'tflite model floating output: {floating_output}', level='debug')

    # Get I/O indices.
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    # If model has int I/O get quantization information.
    if not floating_input:
        input_scale, input_zero_point = input_details[0]['quantization']
    if not floating_output:
        output_scale, output_zero_point = output_details[0]['quantization']

    # Normalize input with its mean and aggregate std used during training.
    train_agg_mean = common.params_appliance[app]['train_agg_mean']
    agg_mean = common.ALT_AGGREGATE_MEAN if common.USE_ALT_STANDARDIZATION else train_agg_mean
    train_agg_std = common.params_appliance[app]['train_agg_std']
    agg_std = common.ALT_AGGREGATE_STD if common.USE_ALT_STANDARDIZATION else train_agg_std
    logger.log(f'Input norm params mean: {agg_mean}, std: {agg_std}', level='debug')
    data = (data - agg_mean) / agg_std

    # Convert input type.
    if not floating_input:
        data = data / input_scale + input_zero_point
        data = data.astype(np.int8)
    else:
        data = data.astype(np.float32)

    # Expand data dimensions to match model input shape.
    # Starting shape = (WINDOW_LENGTH,)
    # Desired shape = (1, WINDOW_LENGTH, 1)
    data = data[np.newaxis, :, np.newaxis]

    # Actually run inference.
    interpreter.set_tensor(input_index, data)
    interpreter.invoke()
    result = interpreter.get_tensor(output_index)
    pred = np.squeeze(result)

    # Convert output type.
    if not floating_output:
        pred = (pred - output_zero_point) * output_scale

    return pred

def get_arduino_data(port) -> list:
    """Get voltage and power from Arduino and timestamp."""
    # Get bytes from Arduino.
    ser_bytes = port.readline()
    # Decode them as utf-8.
    decoded_bytes = ser_bytes.decode('utf-8').rstrip()
    # Split into individual elements and convert to float.
    # Elements are:
    #   rms voltage,
    #   rms current, real power, apparent power for phase 0,
    #   rms current, real power, apparent power for phase 1
    data = [float(d) for d in decoded_bytes.split(',')]
    # Insert UTC datetime.
    data.insert(0, datetime.now(pytz.utc))
    # Insert timestamp.
    data.insert(1, round(time(), 2))
    logger.log(f'Auduino data: {data}', level='debug')
    return data

def get_arguments():
    """Command line arguments."""
    parser = argparse.ArgumentParser(
        description='Read data from arduino and perform inference using raspberry pi.')
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/lindos/nilm/ml/models/',
        help='tflite model path'
    )
    parser.add_argument(
        '--csv_file_name',
        type=str,
        default='/mnt/usbstorage/nilm/garage/samples.csv',
        help='full path of csv file to store samples and predictions'
    )
    parser.add_argument(
        '--apply_threshold',
        action='store_true',
        help='threshold appliance predictions'
    )
    parser.set_defaults(apply_threshold=False)
    return parser.parse_args()

def handler(signal_received, frame):
    """SIGINT handler."""
    print('SIGINT or CTRL-C detected.')
    sys.exit(0)

if __name__ == '__main__':
    logger.log(f'Machine name: {socket.gethostname()}')
    args = get_arguments()
    logger.log('Arguments: ')
    logger.log(args)

    # Run the handler() function when SIGINT is received.
    signal(SIGINT, handler)

    for appliance in APPLIANCES:
        model_filepath = os.path.join(
            args.model_path, appliance, f'{appliance}_{MODEL_ARCH}_{QUANT_MODE}.tflite'
        )
        logger.log(f'Using model {model_filepath} for {appliance}.')

    logger.log('Running. Press CTRL-C to exit.')

    with serial.Serial(SER_PORT, SER_BAUDRATE, timeout=1) as ser:
        if ser.isOpen():
            sleep(1.0) # wait for port to be ready
            logger.log(f'Connected to port {ser.port}.')
            with open(args.csv_file_name, 'w', encoding='utf-8') as csv_file:
                logger.log(f'Opened {args.csv_file_name} for writing samples.')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(CSV_ROW_NAMES)
                ser.reset_input_buffer()

                # Initialize a ndarray to hold windowed mains power readings.
                mains_power = np.empty((0), dtype=np.float32)
                # Initialize sample counter. The sampling rate is set by the Arduino code.
                # E.g., 5 days of samples is 5 * 24 * 60 * 60 / 8 @ 8 sec sampling period.
                sample_num = 0
                logger.log(f'Sample number {sample_num}', level='debug')
                logger.log('Gathering initial window of samples...')
                while True:
                    if ser.in_waiting > 0:
                        # Get mains power data from Arduino.
                        sample = get_arduino_data(ser)
                        # Sum real powers and add to mains window.
                        total_power = sample[4] + sample[7]
                        logger.log(f'Total real power: {total_power:.3f} Watts.', level='debug')
                        # Add sample to window.
                        mains_power = np.append(mains_power, total_power)
                        logger.log(f'Window length: {mains_power.size}', level='debug')

                        if mains_power.size == WINDOW_LENGTH:
                            logger.log(
                                f'Got full window of {len(mains_power)} samples '
                                f'at sample number {sample_num}. '
                                f'Running inference on windowed data...'
                            )
                            # Run inference when enough samples are captured to fill a window.
                            # A single inference takes about 100 ms on this machine from cold start.
                            # This is much faster than sample period (default 8 s) so running
                            # inference in same thread with data capture should not cause dropped data.
                            # Note that separate models are used for each appliance and so cold start
                            # latency will be incurred for each prediction. The models should be combined
                            # so cold start latency is incurred only on the first prediction. This will be
                            # done at some point in the future.
                            start = time()
                            predictions = {
                                appliance : infer(
                                    appliance, mains_power, args.model_path
                                ) for appliance in APPLIANCES
                            }
                            end = time()
                            logger.log('Inference run complete.')
                            num_valid_predictions = np.count_nonzero(
                                ~np.isnan(list(predictions.values()))
                            )
                            logger.log(
                                f'Inference rate: '
                                f'{num_valid_predictions / (end - start):.3f} Hz'
                            )

                            # Post-processing.
                            # De-normalize predictions to get absolute power and threshold.
                            logger.log('Post-processing predictions.')
                            for appliance in APPLIANCES:
                                prediction = predictions[appliance]
                                # If there is no prediction, skip and mark as NaN.
                                if np.isnan(prediction):
                                    logger.log(f'{appliance} has no prediction.')
                                    sample.append(prediction)
                                    continue
                                # De-normalize.
                                mean, std = get_scaling(appliance)
                                logger.log(f'Appliance mean: {mean}', level='debug')
                                logger.log(f'Appliance std: {std}', level='debug')
                                prediction = prediction * std + mean
                                # Zero out negative power predictions.
                                prediction = 0 if prediction < 0 else prediction
                                # Apply on-power threshold.
                                if args.apply_threshold:
                                    appliance_threshold = common.params_appliance[appliance]['on_power_threshold']
                                    logger.log(f'Appliance threshold: {appliance_threshold}', level='debug')
                                    prediction = 0 if prediction < appliance_threshold else prediction
                                logger.log(f'Prediction for {appliance}: {prediction:.3f} Watts.')
                                sample.append(prediction)

                            # Remove oldest sample from window to make room for newest.
                            # TODO: make slice object a variable to adjust output resolution.
                            mains_power = mains_power[1:]
                        else:
                            # At this point there are no predictions, mark them NaN.
                            for appliance in APPLIANCES:
                                sample.append(np.NaN)

                        # Write sample and predictions to csv file.
                        csv_writer.writerow(sample)
                        sample_num +=1

                    sleep(0.01) # avoids maxing out cpu waiting for Arduino data

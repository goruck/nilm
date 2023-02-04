"""
Read data from arduino and perform appliance power inference.

Copyright (c) 2022 Lindo St. Angel.
"""

import os
import argparse
import socket
import sys
import serial
import csv
import pytz
from datetime import datetime
from time import time, sleep
from signal import signal, SIGINT

import tflite_runtime.interpreter as tflite
import numpy as np

sys.path.append('../ml')
from logger import log
from common import params_appliance

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

def infer(appliance: str, input: np.ndarray, args) -> np.ndarray:
    """Perform inference using a tflite model."""
    log(f'Predicting power for {appliance}.', level='debug')

    model_filepath = os.path.join(
        args.model_path, appliance, f'{appliance}_quant.tflite')

    # Start the tflite interpreter.
    try:
        interpreter = tflite.Interpreter(model_path=model_filepath)
        log(f'tflite model: {model_filepath}', level='debug')
    except ValueError as e:
        log(f'{appliance} tflite model not found', level='error')
        log(e, level='error')
        return np.NaN

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    log(f'interpreter input details: {input_details}', level='debug')
    output_details = interpreter.get_output_details()
    log(f'interpreter output details: {output_details}', level='debug')

    # Check I/O tensor type.
    floating_input = input_details[0]['dtype'] == np.float32
    log(f'tflite model floating input: {floating_input}', level='debug')
    floating_output = output_details[0]['dtype'] == np.float32
    log(f'tflite model floating output: {floating_output}', level='debug')

    # Get I/O indices.
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    # If model has int I/O get quantization information.
    if not floating_input:
        input_scale, input_zero_point = input_details[0]['quantization']
    if not floating_output:
        output_scale, output_zero_point = output_details[0]['quantization']

    # Normalize input with its mean and aggregate std used during training.
    input_mean = np.mean(input)
    train_agg_std = params_appliance[appliance]['train_agg_std']
    log(f'Input norm params mean: {input_mean}, std: {train_agg_std}', level='debug')
    input = (input - input_mean) / train_agg_std

    # Convert input type.
    if not floating_input:
        input = input / input_scale + input_zero_point
        input = input.astype(np.int8)
    else:
        input = input.astype(np.float32)

    # Expand input dimensions to match model InputLayer shape.
    # Starting shape = (WINDOW_LENGTH,)
    # Desired shape = (1, 1, WINDOW_LENGTH, 1)
    input = np.expand_dims(input, axis=(0, 1, 3))

    # Actually run inference.
    interpreter.set_tensor(input_index, input)
    interpreter.invoke()
    result = interpreter.get_tensor(output_index)
    prediction = np.squeeze(result)

    # Convert output type.
    if not floating_output:
        prediction = (prediction - output_zero_point) * output_scale

    return prediction

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
    sample = [float(d) for d in decoded_bytes.split(',')]
    # Insert UTC datetime.
    sample.insert(0, datetime.now(pytz.utc))
    # Insert timestamp.
    sample.insert(1, round(time(), 2))
    log(f'Sample = {sample}', level='debug')
    return sample

def get_arguments():
    parser = argparse.ArgumentParser(
        description=f'Read data from arduino and perform '
        f'inference using raspberry pi.')
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/pi/nilm/ml/models/',
        help='tflite model path')
    parser.add_argument(
        '--csv_file_name',
        type=str,
        default='/mnt/usbstorage/nilm/garage/samples.csv',
        help='full path of csv file to store samples and predictions')
    parser.add_argument(
        '--apply_threshold', action='store_true',
        help='threshold appliance predictions')
    parser.set_defaults(apply_threshold=False)
    return parser.parse_args()

def handler(signal_received, frame):
    print('SIGINT or CTRL-C detected.')
    sys.exit(0)

if __name__ == '__main__':
    log(f'Machine name: {socket.gethostname()}')
    args = get_arguments()
    log('Arguments: ')
    log(args)

    # Run the handler() function when SIGINT is received.
    signal(SIGINT, handler)

    for appliance in APPLIANCES:
        model_filepath = os.path.join(
            args.model_path, appliance, f'{appliance}_quant.tflite')
        log(f'Using model {model_filepath} for {appliance}.')

    log('Running. Press CTRL-C to exit.')

    with serial.Serial(SER_PORT, SER_BAUDRATE, timeout=1) as ser:
        if ser.isOpen():
            sleep(1.0) # wait for port to be ready
            log(f'Connected to port {ser.port}.')
            with open(args.csv_file_name, 'w') as csv_file:
                log(f'Opened {args.csv_file_name} for writing samples.')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(CSV_ROW_NAMES)
                ser.reset_input_buffer()

                # Initialize a ndarray to hold windowed mains power readings.
                mains_power = np.empty((0), dtype=np.float32)
                # Initialize sample counter. The sampling rate is set by the Arduino code.
                # E.g., 5 days of samples is 5 * 24 * 60 * 60 / 8 @ 8 sec sampling period.
                sample_num = 0
                log(f'Sample number {sample_num}', level='debug')
                log('Gathering initial window of samples...')
                while True:
                    if ser.in_waiting > 0:
                        # Get mains power data from Arduino.
                        sample = get_arduino_data(ser)
                        # Sum real powers and add to mains window.
                        total_power = sample[4] + sample[7]
                        log(f'Total real power: {total_power:.3f} Watts.', level='debug')
                        # Add sample to window.
                        mains_power = np.append(mains_power, total_power)
                        log(f'Window length: {mains_power.size}', level='debug')

                        if mains_power.size == WINDOW_LENGTH:
                            log(
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
                                    appliance, mains_power, args
                                ) for appliance in APPLIANCES
                            }
                            end = time()
                            log('Inference run complete.')
                            num_valid_predictions = np.count_nonzero(
                                ~np.isnan(list(predictions.values())))
                            log(
                                f'Inference rate: '
                                f'{num_valid_predictions / (end - start):.3f} Hz'
                            )

                            # Post-processing.
                            # De-normalize predictions to get absolute power and threshold.
                            log('Post-processing predictions.')
                            for appliance in APPLIANCES:
                                prediction = predictions[appliance]
                                # If there is no prediction, skip and mark as NaN.
                                if prediction == np.NaN:
                                    log(f'{appliance} has no prediction.')
                                    sample.append(prediction)
                                    continue
                                # De-normalize.
                                mean = params_appliance[appliance]['train_app_mean']
                                std = params_appliance[appliance]['train_app_std']
                                log(f'Appliance mean: {mean}', level='debug')
                                log(f'Appliance std: {std}', level='debug')
                                prediction = prediction * std + mean
                                # Zero out negative power predictions.
                                prediction = 0 if prediction < 0 else prediction
                                # Apply on-power threshold.
                                if args.apply_threshold:
                                    appliance_threshold = params_appliance[appliance]['on_power_threshold']
                                    log(f'Appliance threshold: {appliance_threshold}', level='debug')
                                    prediction = 0 if prediction < appliance_threshold else prediction
                                log(f'Prediction for {appliance}: {prediction:.3f} Watts.')
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
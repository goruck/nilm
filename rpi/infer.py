"""
Read data from arduino and perform appliance power inference.

Copyright (c) 2022 Lindo St. Angel.
"""

import os
import argparse
import socket
import time
import sys
import datetime
import serial
import csv
from signal import signal, SIGINT

import tflite_runtime.interpreter as tflite
import numpy as np

sys.path.append('../ml/')
import nilm_metric as nm
from logger import log

SER_PORT = '/dev/ttyACM0' # rpi serial port where the Arduino is connected
SER_BAUDRATE = 115200 # rpi serial port baudrate

# Row names of csv file that stores power readings and predictions. 
CSV_ROW_NAMES = [
    'DT','TS', # datetime, unix timestamp(UTC)
    'V', # rms main voltage
    'I1','W1','VA1', # mains phase 1 rms current, real power, apparent power
    'I2','W2','VA2', # mains phase 2 rms current, real power, apparent power
    'KP','FP', # kettle, fridge predicted power
    'MP','WP','DP' # microwave, washing machine, dishwasher predected power
]

# These are the appliance types that the models will predict power for.
APPLIANCES = [
    'kettle', 'fridge', 'microwave', 'washingmachine', 'dishwasher'
]

WINDOW_LENGTH = 599 # number of power mains samples to run inference on

params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000},
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800},
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400},
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000},
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700}
    }

def infer(appliance, input, args):
    """Perform inference using a tflite model."""

    # Get tflite model file path.
    model_filepath = os.path.join(args.model_path, appliance,
        f'{appliance}_quant.tflite')
    log(f'tflite model: {model_filepath}')

    # Start the tflite interpreter.
    try:
        interpreter = tflite.Interpreter(model_path=model_filepath)
        log(f'Predicting power for {appliance}.')
    except ValueError:
        log(f'{appliance}_quant.tflite not found')
        return np.NaN

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    log(f'interpreter input details: {input_details}')
    output_details = interpreter.get_output_details()
    log(f'interpreter output details: {output_details}')

    # Add batch dimension to input and convert to ndarray.
    sample = np.expand_dims(input, axis=0)

    # Check I/O tensor type.
    floating_input = input_details[0]['dtype'] == np.float32
    #log(f'tflite model floating input: {floating_input}')
    floating_output = output_details[0]['dtype'] == np.float32
    #log(f'tflite model floating output: {floating_output}')
    # Get I/O indices.
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    # If model has int I/O get quantization information.
    if not floating_input:
        input_scale, input_zero_point = input_details[0]['quantization']
    if not floating_output:
        output_scale, output_zero_point = output_details[0]['quantization']

    if not floating_input: # convert to float to int8
        sample = sample / input_scale + input_zero_point
        sample = sample.astype(np.int8)
    interpreter.set_tensor(input_index, sample)
    interpreter.invoke() # run inference
    result = interpreter.get_tensor(output_index)
    prediction = np.squeeze(result)
    if not floating_output: # convert int8 to float
        prediction = (prediction - output_zero_point) * output_scale

    return prediction

def get_arduino_data(port):
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
    # Insert datetime.
    sample.insert(0, datetime.datetime.now())
    # Insert timestamp.
    sample.insert(1, round(time.time(), 2))
    #print(f'sample = {sample}')
    return sample

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Read data from arduino and perform inference using raspberry pi.')
    parser.add_argument('--sample_max',
                        type=int,
                        default=int(5 * 24 * 60 * 60 / 8), # 5 days @ 8 sec sampling period
                        help='maximum number of mains samples to capture')
    parser.add_argument('--model_path',
                        type=str,
                        default='/home/pi/nilm/ml/models/',
                        help='tflite model path')
    parser.add_argument('--csv_file_name',
                        type=str,
                        default='/mnt/usbstorage/nilm/garage/samples.csv',
                        help='full path of csv file to store samples and predictions')
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

    log('Running. Press CTRL-C to exit.')

    with serial.Serial(SER_PORT, SER_BAUDRATE, timeout=1) as ser:
        if ser.isOpen():
            time.sleep(1.0) # wait for port to be ready
            log(f'Connected to port {ser.port}.')
            with open(args.csv_file_name, 'w') as csv_file:
                log(f'Opened {args.csv_file_name} for writing {args.sample_max} samples.')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(CSV_ROW_NAMES)
                ser.reset_input_buffer()

                mains_power = [] # a windows length size of mains power readings.
                sample_num = 0
                while sample_num < args.sample_max:
                    if ser.in_waiting > 0:
                        # Get mains power data from Arduino.
                        sample = get_arduino_data(ser)
                        # Sum apparent powers and add to mains window.
                        total_power = sample[5] + sample[8]
                        mains_power.append(total_power)
                        #print(f'window length: {len(mains_power)}')

                        if len(mains_power) == WINDOW_LENGTH:
                            # Run inference when enough samples are captured to fill a window.
                            # A single inference takes about 100 ms on this machine from cold start.
                            # This is much faster than sample period (default 8 s) so running
                            # inference in same thread with data capture should not cause dropped data.
                            # Note that separate models are used for each appliance and so cold start
                            # latency will be incurred for each prediction. The models should be combined
                            # so cold start latency is incurred only on the first prediction. This will be
                            # done at some point in the future. 
                            start = time.time()
                            predictions = {
                                appliance : infer(
                                    appliance, mains_power, args
                                ) for appliance in APPLIANCES
                            }
                            end = time.time()
                            log('Inference run(s) complete.')
                            num_valid_predictions = np.count_nonzero(
                                ~np.isnan(list(predictions.values())))
                            log(f'Inference rate: {num_valid_predictions / (end - start):.3f} Hz')

                            # Clear mains power window.
                            mains_power = []

                            for appliance in APPLIANCES:
                                log(f'Post-processing prediction for {appliance}.')
                                prediction = predictions[appliance]
                                # If there is no prediction, skip and mark as NaN.
                                if prediction == np.NaN:
                                    log('Appliance has no prediction.')
                                    sample.append(prediction)
                                    continue
                                # De-normalize.
                                mean = params_appliance[appliance]['mean']
                                std = params_appliance[appliance]['std']
                                log(f'Appliance mean: {mean}')
                                log(f'Appliance std: {std}')
                                prediction = prediction * std + mean
                                # Apply on-power threshold.
                                appliance_threshold = params_appliance[appliance]['on_power_threshold']
                                log(f'Appliance threshold: {appliance_threshold}')
                                prediction = 0 if prediction < appliance_threshold else prediction
                                sample.append(prediction)
                        else:
                            # At this point there are no predictions, mark them NaN.
                            for appliance in APPLIANCES:
                                sample.append(np.NaN)
                        
                        # Write sample and predictions to csv file.
                        csv_writer.writerow(sample)
                        sample_num +=1

                    time.sleep(0.01) # avoids maxing out cpu wating for Arduino data
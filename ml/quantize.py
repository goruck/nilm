"""
Convert a trained Keras model to tflite and quantize it.

Copyright (c) 2022 Lindo St. Angel.
"""

from functools import partial
import os
import argparse
import socket
import shutil

import tensorflow as tf
import numpy as np

from common import load_dataset, WindowGenerator, params_appliance
from logger import log

# Temporary file name to store a model used in this process.
TMP_MODEL_FILE = '/tmp/new_model'

# Number of samples used for post-training quantization.
NUM_CAL = 200

def change_model_batch_shape(model, batch_shape):
    """
    Change input model's batch shape.

    Ref: https://discuss.tensorflow.org/t/change-batch-size-statically-for-inference-tf2/5995
    """
    model_config = model.get_config()
    model_config['layers'][0] = {
        'name': 'new_input',
        'class_name': 'InputLayer',
        'config': {
            'batch_input_shape': batch_shape,
            'dtype': 'float32',
            'sparse': False,
            'name': 'modified_input'
        },
        'inbound_nodes': []
    }
    model_config['layers'][1]['inbound_nodes'] = [[['new_input', 0, 0, {}]]]
    model_config['input_layers'] = [['new_input', 0, 0]]
    new_model = model.__class__.from_config(model_config, custom_objects={})
    new_model.set_weights(model.get_weights())
    return new_model

def representative_dataset_gen(provider, batch_size, num_cal) -> np.float32:
    """TBA"""
    # Get a random batch number.
    rng = np.random.default_rng()
    batch_index = rng.choice(batch_size)

    # Get a random batch of samples.
    samples = provider.__getitem__(batch_index)
    rng.shuffle(samples)

    # Grab a random num_cal sized chunk of those samples.
    samples = rng.choice(samples, size=num_cal)

    for sample in samples:
        # Expand dimension creates a single batch.
        sample = np.expand_dims(sample, axis=0)
        yield [sample]

def convert(model_dir, provider, batch_size, num_cal):
    """TBA"""
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # ref_gen must be a callable so use partial to set parameters. 
    # Alternatively a lambda could be used here.
    # Ref: https://stackoverflow.com/questions/49280016/how-to-make-a-generator-callable
    ref_gen = partial(representative_dataset_gen,
        provider=provider,
        batch_size=batch_size,
        num_cal=num_cal)
    converter.representative_dataset = tf.lite.RepresentativeDataset(ref_gen)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    return converter.convert()

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Convert a trained Keras model to tflite and quantize.')
    parser.add_argument('--appliance_name',
                        type=str,
                        default='kettle',
                        help='the name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default='/home/lindo/Develop/nilm/ml/dataset_management/refit',
                        help='this is the directory of the calibration samples')
    parser.add_argument('--save_dir',
                        type=str,
                        default='/home/lindo/Develop/nilm/ml/models',
                        help='this is the directory to save the quantized model')
    parser.add_argument('--batchsize',
                        type=int,
                        default=1000,
                        help='The batch size of training examples')
    parser.add_argument('--crop',
                        type=int,
                        default=None,
                        help='Partial number of cal samples to use. Default uses entire dataset.')
    return parser.parse_args()

if __name__ == '__main__':
    log(f'Machine name: {socket.gethostname()}')
    args = get_arguments()
    log('Arguments: ')
    log(args)

    # The appliance to train on.
    appliance_name = args.appliance_name

    # Path for calibration data.
    cal_path = os.path.join(
        args.datadir,appliance_name,f'{appliance_name}_training_.csv')
    log(f'Calibration dataset: {cal_path}')

    # offset parameter from window length
    offset = int(0.5 * (params_appliance[appliance_name]['windowlength'] - 1.0))

    window_length = params_appliance[appliance_name]['windowlength']

    model_filepath = os.path.join(args.save_dir, appliance_name)
    log(f'Model file path: {model_filepath}')
    checkpoint_filepath = os.path.join(model_filepath,'checkpoints')
    log(f'Checkpoint file path: {checkpoint_filepath}')

    # Load model checkpoint and change batch shape to (1, window_length).
    # This will make the batch size static for use in downstream processing.
    # The edge tpu compiler only accepts static batch sizes.
    model = tf.keras.models.load_model(checkpoint_filepath)
    model.summary()
    new_model = change_model_batch_shape(model, batch_shape=(1, window_length))
    new_model.summary()
    tf.keras.models.save_model(new_model, filepath=TMP_MODEL_FILE)

    # Load datasets.
    dataset = load_dataset(cal_path, args.crop)

    num_train_samples = dataset[0].size
    log(f'There are {num_train_samples/10**6:.3f}M samples in dataset.')

    dataset_provider = WindowGenerator(
        dataset=dataset,
        offset=offset,
        batch_size=args.batchsize,
        train=False)

    # Convert model to tflite and quantize.
    # Note: Input and output layers will be converted to uint8.
    tflite_model_quant = convert(
        model_dir=TMP_MODEL_FILE,
        provider=dataset_provider,
        batch_size=args.batchsize,
        num_cal=NUM_CAL)

    # Save converted model.
    filepath = os.path.join(args.save_dir, appliance_name,
        f'{appliance_name}_quant.tflite') 
    with open(filepath, 'wb') as file:
        file.write(tflite_model_quant)
    log(f'Quantized tflite model saved to {filepath}.')

    # No need to keep the Keras model with the static batch size.
    # So delete it.
    try:
        shutil.rmtree(TMP_MODEL_FILE)
    except OSError as e:
        print(f'Error: {e.filename} {e.strerror}')
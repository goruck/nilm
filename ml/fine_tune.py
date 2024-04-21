"""Fine tune a neural network to perform energy disaggregation.

Improve performance of a NILM model by fine tuning it with locally obtained
ground truth appliance data. The model is assumed to have been trained from
scratch using large datasets from a variety of houses and appliances not
dissimilar to the ones being used locally.

Use ml/train.py for training the model from scratch. 

Copyright (c) 2024 Lindo St. Angel
"""

import os
import argparse
import socket

import matplotlib.pyplot as plt
import tensorflow as tf

from logger import Logger
import common
from distributed_trainer import DistributedTrainer

def smooth_curve(points, factor=0.8):
    """Smooth a series of points given a smoothing factor."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot(data, plot_name, plot_display, appliance, save_dir):
    """Save and display loss and mae plots."""
    loss = data['loss']
    val_loss = data['val_loss']
    plt.plot(
        [i[0] for i in loss],
        smooth_curve([i[1] for i in loss]),
        label='Smoothed Training Loss'
    )
    plt.plot(
        [i[0] for i in val_loss],
        smooth_curve([i[1] for i in val_loss]),
        label='Smoothed Validation Loss'
    )
    plt.title(f'Training history for {appliance} ({plot_name})')
    plt.ylabel('Loss')
    plt.xlabel('Training Step')
    plt.legend()
    plot_filepath = os.path.join(save_dir, appliance, f'{plot_name}_loss')
    plt.savefig(fname=plot_filepath)
    if plot_display:
        plt.show()
    plt.close()
    # Mean Absolute Error.
    val_mae = data['val_mae']
    plt.plot([i[0] for i in val_mae], smooth_curve([i[1] for i in val_mae]))
    plt.title(f'Smoothed validation MAE for {appliance} ({plot_name})')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Training Step')
    plot_filepath = os.path.join(save_dir, appliance, f'{plot_name}_mae')
    plt.savefig(fname=plot_filepath)
    if plot_display:
        plt.show()
    plt.close()

def get_arguments():
    parser = argparse.ArgumentParser(
        description=(
            'Fine tune a neural network for energy disaggregation -'
            'network input = mains window; network target = the states of '
            'the target appliance.'
        )
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
        help='Network architecture to use'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='./dataset_management/my-house',
        help='Directory of the training samples.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/home/lindo/Develop/nilm/ml/models',
        help='Directory to save the trained models and checkpoints.'
    )
    parser.add_argument(
        '--batchsize',
        type=int,
        default=512,
        help='mini-batch size'
    )
    parser.add_argument(
        '--n_epoch',
        type=int,
        default=50,
        help='Number of epochs to train over.'
    )
    parser.add_argument(
        '--crop_train_dataset',
        type=int,
        default=None,
        help='Number of train samples to use. Default uses entire dataset.'
    )
    parser.add_argument(
        '--crop_val_dataset',
        type=int,
        default=None,
        help='Number of val samples to use. Default uses entire dataset.'
    )
    parser.add_argument(
        '--plot_display',
        action='store_true',
        help='Display loss and accuracy curves.'
    )
    parser.set_defaults(plot_display=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()

    # The appliance to train on.
    appliance_name = args.appliance_name

    logger = Logger(
        log_file_name=os.path.join(
            args.save_dir,
            appliance_name,
            f'{appliance_name}_fine_tune_{args.model_arch}.log'
        )
    )

    logger.log(f'Machine name: {socket.gethostname()}')
    logger.log('Arguments: ')
    logger.log(args)

    window_length = common.params_appliance[appliance_name]['window_length']
    logger.log(f'Window length: {window_length}')

    # Look for the training set.
    for filename in os.listdir(os.path.join(args.datadir, appliance_name)):
        if 'training' in filename:
            training_filename = filename
    # Path for training data.
    training_path = os.path.join(args.datadir, appliance_name, training_filename)
    logger.log(f'Training dataset: {training_path}')

    # Look for the validation set.
    for filename in os.listdir(os.path.join(args.datadir, appliance_name)):
        if 'validation' in filename:
            val_filename = filename
    # Path for validation data.
    validation_path = os.path.join(args.datadir,appliance_name, val_filename)
    logger.log(f'Validation dataset: {validation_path}')

    model_filepath = os.path.join(args.save_dir, appliance_name)
    restore_model_filepath = os.path.join(model_filepath, f'savemodel_{args.model_arch}')
    logger.log(f'Restoring model from: {restore_model_filepath}')

    checkpoint_filepath = os.path.join(model_filepath, f'checkpoints_{args.model_arch}_fine_tune')
    logger.log(f'Checkpoint file path: {checkpoint_filepath}')

    savemodel_filepath = os.path.join(model_filepath, f'savemodel_{args.model_arch}_fine_tune')
    logger.log(f'SaveModel file path: {savemodel_filepath}')

    history_filepath = os.path.join(model_filepath, f'history_{args.model_arch}_fine_tune')
    logger.log(f'Training history file path: {history_filepath}')

    # Load datasets.
    train_dataset = common.load_dataset(training_path, args.crop_train_dataset)
    val_dataset = common.load_dataset(validation_path, args.crop_train_dataset)
    logger.log(f'There are {train_dataset[0].size/10**6:.3f}M training samples.')
    logger.log(f'There are {val_dataset[0].size/10**6:.3f}M validation samples.')

    # Calculate normalized threshold for appliance status determination.
    threshold = common.params_appliance[appliance_name]['on_power_threshold']
    max_on_power = common.params_appliance[appliance_name]['max_on_power']
    threshold /= max_on_power
    logger.log(f'Normalized on power threshold: {threshold}')

    # Get L1 loss multiplier.
    c0 = common.params_appliance[appliance_name]['c0']
    logger.log(f'L1 loss multiplier: {c0}')

    def fine_tune_model(num_classifier_layers=3):
        """Create a model for fine tuning."""
        base_model = tf.keras.models.load_model(restore_model_filepath)
        # Freeze all layers in base model
        base_model.trainable = False
        # Remove base model's classifier layers which leaves a feature extractor.
        feature_extractor = base_model.layers[:-num_classifier_layers]
        classifier = [
            tf.keras.layers.Dense(1024, activation='relu', name='dense1'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation='relu', name='dense2'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='linear', name='dense3')
        ]
        # Return the base model with new, trainable output layers.
        return tf.keras.Sequential(feature_extractor+classifier, name='fine_tune')

    trainer = DistributedTrainer(
        do_not_use_distributed_training=False,
        resume_training=False,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batchsize,
        model_fn=fine_tune_model,
        window_length=window_length,
        checkpoint_filepath=checkpoint_filepath,
        logger=logger
    )

    train_history = trainer.train(
        epochs=args.n_epoch,
        threshold=threshold,
        c0=c0,
        savemodel_filepath=savemodel_filepath,
        history_filepath=history_filepath
    )

    plot(
        train_history,
        plot_name=f'fine_tune_{args.model_arch}',
        plot_display=args.plot_display,
        appliance=args.appliance_name,
        save_dir=args.save_dir
    )

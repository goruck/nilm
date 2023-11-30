"""Train a neural network to perform energy disaggregation.

Given a sequence of electricity mains reading, the algorithm
separates the mains into appliances.

Copyright (c) 2022~2023 Lindo St. Angel
"""

import os
import argparse
import socket

import matplotlib.pyplot as plt

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

def plot(data, plot_name, plot_display, appliance, save_dir, logger):
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
    logger.log(f'Plot directory: {plot_filepath}')
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
    logger.log(f'Plot directory: {plot_filepath}')
    plt.savefig(fname=plot_filepath)
    if plot_display:
        plt.show()
    plt.close()

def get_arguments():
    parser = argparse.ArgumentParser(
        description=(
            'Train a neural network for energy disaggregation -'
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
        default='./dataset_management/refit',
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
        '--do_not_use_distributed_training',
        action='store_true',
        help='Use only GPU 0 for training.'
    )
    parser.add_argument(
        '--resume_training',
        action='store_true',
        help='Resume training from last checkpoint.'
    )
    parser.add_argument(
        '--plot_display',
        action='store_true',
        help='Display loss and accuracy curves.'
    )
    parser.set_defaults(do_not_use_distributed_training=False)
    parser.set_defaults(resume_training=False)
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
            f'{appliance_name}_train_{args.model_arch}.log'
        ),
        append=args.resume_training # append rest of training to end of existing log
    )

    logger.log(
        '*** Resuming training from last checkpoint ***' if args.resume_training
        else '*** Training model from scratch ***'
    )
    logger.log(f'Machine name: {socket.gethostname()}')
    logger.log('Arguments: ')
    logger.log(args)

    window_length = common.params_appliance[appliance_name]['window_length']
    logger.log(f'Window length: {window_length}')

    # Path for training data.
    training_path = os.path.join(
        args.datadir, appliance_name, f'{appliance_name}_training_.csv'
    )
    logger.log(f'Training dataset: {training_path}')

    # Look for the validation set.
    for filename in os.listdir(os.path.join(args.datadir, appliance_name)):
        if 'validation' in filename:
            val_filename = filename
    # Path for validation data.
    validation_path = os.path.join(args.datadir,appliance_name, val_filename)
    logger.log(f'Validation dataset: {validation_path}')

    model_filepath = os.path.join(args.save_dir, appliance_name)
    checkpoint_filepath = os.path.join(model_filepath, f'checkpoints_{args.model_arch}')
    logger.log(f'Checkpoint file path: {checkpoint_filepath}')
    savemodel_filepath = os.path.join(model_filepath, f'savemodel_{args.model_arch}')
    logger.log(f'SaveModel file path: {savemodel_filepath}')

    # Load datasets.
    train_dataset = common.load_dataset(training_path, args.crop_train_dataset)
    val_dataset = common.load_dataset(validation_path, args.crop_val_dataset)
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

    trainer = DistributedTrainer(
        do_not_use_distributed_training=args.do_not_use_distributed_training,
        resume_training=args.resume_training,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batchsize,
        model_arch=args.model_arch,
        window_length=window_length,
        checkpoint_filepath=checkpoint_filepath,
        logger=logger
    )

    train_history = trainer.train(
        epochs=args.n_epoch,
        threshold=threshold,
        c0=c0,
        savemodel_filepath=savemodel_filepath
    )

    plot(
        train_history,
        plot_name=f'train_{args.model_arch}',
        plot_display=args.plot_display,
        appliance=args.appliance_name,
        save_dir=args.save_dir,
        logger=logger
    )

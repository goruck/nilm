"""Train a neural network to perform energy disaggregation.

Given a sequence of electricity mains reading, the algorithm
separates the mains into appliances.

References:
(1) Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton.
``Sequence-to-point learning with neural networks for nonintrusive load monitoring."
Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), Feb. 2-7, 2018.

(2) https://arxiv.org/abs/1902.08835

(3) https://github.com/MingjunZhong/transferNILM.

Copyright (c) 2022 Lindo St. Angel
"""

import os
import argparse
import socket

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt

from define_models import create_model
from logger import log
from common import load_dataset, get_window_generator

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

def plot(history, plot_name, plot_display, appliance_name):
    """Save and display mse and mae plots."""
    # Mean square error.
    mse = history.history['mse']
    val_mse = history.history['val_mse']
    plot_epochs = range(1,len(mse)+1)
    plt.plot(
        plot_epochs, smooth_curve(mse),
        label='Smoothed Training MSE')
    plt.plot(
        plot_epochs, smooth_curve(val_mse),
        label='Smoothed Validation MSE')
    plt.title(f'Training history for {appliance_name} ({plot_name})')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend()
    plot_filepath = os.path.join(
        args.save_dir, appliance_name, f'{plot_name}_mse')
    log(f'Plot directory: {plot_filepath}')
    plt.savefig(fname=plot_filepath)
    if plot_display:
        plt.show()
    plt.close()
    # Mean Absolute Error.
    val_mae = history.history['val_mae']
    plt.plot(plot_epochs, smooth_curve(val_mae))
    plt.title(f'Smoothed validation MAE for {appliance_name} ({plot_name})')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plot_filepath = os.path.join(
        args.save_dir, appliance_name, f'{plot_name}_mae')
    log(f'Plot directory: {plot_filepath}')
    plt.savefig(fname=plot_filepath)
    if plot_display:
        plt.show()
    plt.close()

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Train a neural network for energy disaggregation - \
            network input = mains window; network target = the states of \
            the target appliance.')
    parser.add_argument(
        '--appliance_name',
        type=str,
        default='kettle',
        help='the name of target appliance')
    parser.add_argument(
        '--datadir',
        type=str,
        default='./dataset_management/refit',
        help='this is the directory of the training samples')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./models',
        help='this is the directory to save the trained models')
    parser.add_argument(
        '--prune_log_dir',
        type=str,
        default='/home/lindo/Develop/nilm/ml/pruning_logs',
        help='location of pruning logs')
    parser.add_argument(
        '--batchsize',
        type=int,
        default=1000,
        help='The batch size of training examples')
    parser.add_argument(
        '--n_epoch',
        type=int,
        default=50,
        help='The number of epochs.')
    parser.add_argument(
        '--prune_end_epoch',
        type=int,
        default=15,
        help='The number of epochs to prune over.')
    parser.add_argument(
        '--crop_train_dataset',
        type=int,
        default=None,
        help='Number of train samples to use. Default uses entire dataset.')
    parser.add_argument(
        '--crop_val_dataset',
        type=int,
        default=None,
        help='Number of val samples to use. Default uses entire dataset.')
    parser.add_argument(
        '--qat', action='store_true',
        help='Fine-tune pre-trained model with quantization aware training.')
    parser.add_argument(
        '--prune', action='store_true',
        help='Prune pre-trained model for on-device inference.')
    parser.add_argument(
        '--train', action='store_true',
        help='If set, train model from scratch.')
    parser.add_argument(
        '--plot', action='store_true',
        help='If set, display plots.')
    parser.set_defaults(plot=False)
    parser.set_defaults(qat=False)
    parser.set_defaults(prune=False)
    parser.set_defaults(train=False)
    return parser.parse_args()

if __name__ == '__main__':
    log(f'Machine name: {socket.gethostname()}')
    log(f'tf version: {tf.version.VERSION}')
    args = get_arguments()
    log('Arguments: ')
    log(args)

    # The appliance to train on.
    appliance_name = args.appliance_name

    # Path for training data.
    training_path = os.path.join(
        args.datadir,appliance_name,f'{appliance_name}_training_.csv')
    log(f'Training dataset: {training_path}')

    # Look for the validation set
    for filename in os.listdir(os.path.join(args.datadir, appliance_name)):
        if 'validation' in filename:
            val_filename = filename
    # path for validation data
    validation_path = os.path.join(args.datadir,appliance_name,val_filename)
    log(f'Validation dataset: {validation_path}')

    model_filepath = os.path.join(args.save_dir, appliance_name)
    log(f'Model file path: {model_filepath}')

    checkpoint_filepath = os.path.join(model_filepath,'checkpoints')
    log(f'Checkpoint file path: {checkpoint_filepath}')

    # Load datasets.
    train_dataset = load_dataset(training_path, args.crop_train_dataset)
    val_dataset = load_dataset(validation_path, args.crop_val_dataset)
    num_train_samples = train_dataset[0].size
    log(f'There are {num_train_samples/10**6:.3f}M training samples.')
    num_val_samples = val_dataset[0].size
    log(f'There are {num_val_samples/10**6:.3f}M validation samples.')

    # Init window generator to provide samples and targets.
    WindowGenerator = get_window_generator()
    training_provider = WindowGenerator(dataset=train_dataset)
    validation_provider = WindowGenerator(dataset=val_dataset, shuffle=False)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_mse',
        patience=6,
        verbose=2)

    if args.train:
        log('Training model from scratch.')

        model = create_model()

        model.summary()

        # Decay lr at 1/t every 5 epochs.
        batches_per_epoch = training_provider.__len__()
        epochs_per_decay_step = 5
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=batches_per_epoch * epochs_per_decay_step,
            decay_rate=1,
            staircase=False)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08),
            loss='mse',
            metrics=['mse', 'msle', 'mae'])

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = checkpoint_filepath,
            monitor='val_mse',
            verbose=1,
            save_best_only=True,
            mode='auto',
            save_freq='epoch')

        callbacks = [early_stopping, checkpoint_callback]

        history = model.fit(
            x=training_provider,
            steps_per_epoch=None,
            epochs=args.n_epoch,
            callbacks=callbacks,
            validation_data=validation_provider,
            validation_steps=None,
            workers=24,
            use_multiprocessing=True)

        plot(
            history,
            plot_name='train',
            plot_display=args.plot,
            appliance_name=appliance_name)
    elif args.qat:
        log('Fine-tuning pre-trained model with quantization aware training.')

        quantize_model = tfmot.quantization.keras.quantize_model

        model = tf.keras.models.load_model(checkpoint_filepath)

        q_aware_model = quantize_model(model)

        q_aware_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.0001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08),
            loss='mse',
            metrics=['mse', 'msle', 'mae'])

        q_aware_model.summary()

        q_checkpoint_filepath = os.path.join(model_filepath,'qat_checkpoints')
        log(f'QAT checkpoint file path: {q_checkpoint_filepath}')

        q_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = q_checkpoint_filepath,
            monitor='val_mse',
            verbose=1,
            save_best_only=True,
            mode='auto',
            save_freq='epoch')

        callbacks = [early_stopping, q_checkpoint_callback]

        history = q_aware_model.fit(
            x=training_provider,
            steps_per_epoch=None,
            epochs=args.n_epoch,
            callbacks=callbacks,
            validation_data=validation_provider,
            validation_steps=None,
            workers=24,
            use_multiprocessing=True)

        plot(
            history,
            plot_name='qat',
            plot_display=args.plot,
            appliance_name=appliance_name)
    elif args.prune:
        log('Prune pre-trained model for on-device inference.')

        model = tf.keras.models.load_model(checkpoint_filepath)

        # Compute end step to finish pruning after 15 epochs.
        end_step = (num_train_samples // args.batchsize) * args.prune_end_epoch

        # Define parameters for pruning.
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.25,
                final_sparsity=0.75,
                begin_step=0,
                end_step=end_step)
        }

        # Sparsifies the layer's weights during training.
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        # Try to apply pruning wrapper with pruning policy parameter.
        try:
            model_for_pruning = prune_low_magnitude(model, **pruning_params)
        except ValueError as e:
            log(e, level='error')
            exit()

        model_for_pruning.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.0001, # lower rate than training from scratch
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08),
            loss='mse',
            metrics=['mse', 'msle', 'mae'])

        model_for_pruning.summary()

        pruning_checkpoint_filepath = os.path.join(
            model_filepath,'pruning_checkpoints')
        log(f'Pruning checkpoint file path: {pruning_checkpoint_filepath}')

        pruning_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = pruning_checkpoint_filepath,
            monitor='val_mse',
            verbose=1,
            save_best_only=True,
            mode='auto',
            save_freq='epoch')

        pruning_callbacks = [
            pruning_checkpoint_callback,
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=args.prune_log_dir)
        ]

        history = model_for_pruning.fit(
            x=training_provider,
            steps_per_epoch=None,
            epochs=args.prune_end_epoch,
            callbacks=pruning_callbacks,
            validation_data=validation_provider,
            validation_steps=None,
            workers=24,
            use_multiprocessing=True)

        plot(
            history,
            plot_name='prune',
            plot_display=args.plot,
            appliance_name=appliance_name)

        model_for_pruning.summary()

        pruned_model_filepath = os.path.join(model_filepath,'pruned_model')
        log(f'Final pruned model file path: {pruned_model_filepath}')
        model_for_pruning.save(pruned_model_filepath)

        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        model_for_export.summary()

        pruned_model_for_export_filepath = os.path.join(model_filepath,'pruned_model_for_export')
        log(f'Pruned model for export file path: {pruned_model_for_export_filepath}')
        model_for_export.save(pruned_model_for_export_filepath)
    else:
        print('Nothing was done, train, qat or prune must be selected.')
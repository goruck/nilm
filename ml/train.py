"""
Train a neural network to perform energy disaggregation,
i.e., given a sequence of electricity mains reading,
the algorithm separates the mains into appliances.

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
import matplotlib.pyplot as plt

from cnn_model import create_model
from logger import log
from common import load_dataset, WindowGenerator, params_appliance

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def get_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network\
                                     for energy disaggregation - \
                                     network input = mains window; \
                                     network target = the states of \
                                     the target appliance.')
    parser.add_argument('--appliance_name',
                        type=str,
                        default='kettle',
                        help='the name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default='./dataset_management/refit',
                        help='this is the directory of the training samples')
    parser.add_argument('--pretrainedmodel_dir',
                        type=str,
                        default='./pretrained_model',
                        help='this is the directory of the pre-trained models')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./models',
                        help='this is the directory to save the trained models')
    parser.add_argument('--batchsize',
                        type=int,
                        default=1000,
                        help='The batch size of training examples')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=50,
                        help='The number of epochs.')
    parser.add_argument('--save_model',
                        type=int,
                        default=-1,
                        help='Save the learnt model:\
                        0 -- not to save the learnt model parameters;\
                        n (n>0) -- to save the model params every n steps;\
                        -1 -- only save the learnt model params\
                        at the end of training.')
    parser.add_argument('--dense_layers',
                        type=int,
                        default=1,
                        help=':\
                                1 -- One dense layers;\
                                2 -- Two dense layers;\
                                3 -- Three dense layers.')
    parser.add_argument("--transfer_model", action='store_true',
                        help="If set: using entire pre-trained model.\
                             Else: retrain the entire pre-trained model;\
                             This will override the 'transfer_cnn' and 'cnn' parameters;\
                             The appliance_name parameter will use to retrieve \
                             the entire pre-trained model of that appliance.")
    parser.add_argument("--transfer_cnn", action='store_true',
                        help="If set: using a pre-trained CNN\
                              Else: not using a pre-trained CNN.")
    parser.add_argument('--cnn',
                        type=str,
                        default='kettle',
                        help='The CNN trained by which appliance to load (pretrained model).')
    parser.add_argument('--crop_train_dataset',
                        type=int,
                        default=None,
                        help='Partial number of training samples to use. Default uses entire dataset.')
    parser.add_argument('--crop_val_dataset',
                        type=int,
                        default=None,
                        help='Partial number of val samples to use. Default uses entire dataset.')
    parser.add_argument('--show_plot', action='store_true',
                        help='If set, display training result plots.')
    parser.set_defaults(transfer_model=False)
    parser.set_defaults(transfer_cnn=False)
    parser.set_defaults(show_plot=False)
    return parser.parse_args()

if __name__ == '__main__':
    log(f'Machine name: {socket.gethostname()}')
    args = get_arguments()
    log('Arguments: ')
    log(args)

    # The appliance to train on.
    appliance_name = args.appliance_name

    # Path for training data.
    training_path = os.path.join(
        args.datadir,appliance_name,f'{appliance_name}_training_.csv')
    log(f'Training dataset: {training_path}')

    # Looking for the validation set
    for filename in os.listdir(os.path.join(args.datadir,appliance_name)):
        if 'validation' in filename:
            val_filename = filename

    # path for validation data
    validation_path = os.path.join(args.datadir,appliance_name,val_filename)
    log(f'Validation dataset: {validation_path}')

    # offset parameter from window length
    offset = int(0.5 * (params_appliance[appliance_name]['windowlength'] - 1.0))

    window_length = params_appliance[appliance_name]['windowlength']

    model = create_model(
        input_window_length=window_length, n_dense=args.dense_layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08),
        loss='mse',
        metrics=['mse', 'msle', 'mae'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        verbose=2)

    model_filepath = os.path.join(args.save_dir, appliance_name)
    log(f'Model file path: {model_filepath}')
    checkpoint_filepath = os.path.join(model_filepath,'checkpoints')
    log(f'Checkpoint file path: {checkpoint_filepath}')

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch')

    callbacks = [early_stopping, checkpoint_callback]

    # Load datasets.
    train_dataset = load_dataset(training_path, args.crop_train_dataset)
    val_dataset = load_dataset(validation_path, args.crop_val_dataset)

    training_provider = WindowGenerator(
        dataset=train_dataset,
        offset=offset,
        batch_size=args.batchsize)
    validation_provider = WindowGenerator(
        dataset=val_dataset,
        offset=offset,
        batch_size=args.batchsize,
        shuffle=False)

    num_train_samples = train_dataset[0].size
    log(f'There are {num_train_samples/10**6:.3f}M training samples.')
    num_val_samples = val_dataset[0].size
    log(f'There are {num_val_samples/10**6:.3f}M validation samples.')

    history = model.fit(
        x=training_provider,
        steps_per_epoch=None,
        epochs=args.n_epoch,
        callbacks=callbacks,
        validation_data=validation_provider,
        validation_steps=None,
        workers=24,
        use_multiprocessing=True)

    model.summary()

    # Save results and show plots if set.
    # Losses.
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plot_epochs = range(1,len(loss)+1)
    plt.plot(
        plot_epochs, smooth_curve(loss),
        label='Smoothed Training Loss')
    plt.plot(
        plot_epochs, smooth_curve(val_loss),
        label='Smoothed Validation Loss')
    plt.title('Training History')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.xlabel('Epoch')
    plt.legend()
    plot_filepath = os.path.join(
        args.save_dir,appliance_name,'loss.png')
    log(f'Plot directory: {plot_filepath}')
    plt.savefig(fname=plot_filepath)
    if args.show_plot:
        plt.show()
    plt.close()
    # Mean Absolute Error.
    val_mae = history.history['val_mae']
    plt.plot(plot_epochs, smooth_curve(val_mae))
    plt.title('Smoothed Validation MAE')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plot_filepath = os.path.join(
        args.save_dir,args.appliance_name,'mae.png')
    log(f'Plot directory: {plot_filepath}')
    plt.savefig(fname=plot_filepath)
    if args.show_plot:
        plt.show()
    plt.close()
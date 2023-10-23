"""Train a neural network to perform energy disaggregation.

Given a sequence of electricity mains reading, the algorithm
separates the mains into appliances.

Uses distributed GPU training.

Copyright (c) 2022~2023 Lindo St. Angel
"""

import os
import argparse
import socket

import tensorflow as tf
import matplotlib.pyplot as plt

import define_models
from logger import Logger
import common

### DO NOT USE MIXED-PRECISION - CURRENTLY GIVES POOR MODEL ACCURACY ###
# TODO: fix.
# Run in mixed-precision mode for ~30% speedup vs TensorFloat-32
# w/GPU compute capability = 8.6.
USE_MIXED_PRECISION = False
if USE_MIXED_PRECISION:
    from keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

# Set to True run in TF eager mode for debugging.
# May have to reduce batch size <= 512 to avoid OOM.
# Turn off distributed training for best results.
RUN_EAGERLY = False
tf.config.run_functions_eagerly(run_eagerly=RUN_EAGERLY)

# BASE_LR is a good learning rate for a batch size of 1024
# which typically leads to best test accuracy. Other batch
# sizes and learning rates have not been optimized and should
# be avoided for now. TODO: optimize LR & batch sizes.
BASE_LR = 1.0e-4
LR_BY_BATCH_SIZE = {
    512: BASE_LR / tf.sqrt(2.0),
    1024: BASE_LR,
    2048: BASE_LR * tf.sqrt(2.0),
    4096: BASE_LR * tf.sqrt(4.0)}

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

def plot(data, plot_name, plot_display, appliance):
    """Save and display loss and mae plots."""
    loss = data['loss']
    val_loss = data['val_loss']
    plot_epochs = range(1,len(loss)+1)
    plt.plot(
        plot_epochs, smooth_curve(loss),
        label='Smoothed Training Loss')
    plt.plot(
        plot_epochs, smooth_curve(val_loss),
        label='Smoothed Validation Loss')
    plt.title(f'Training history for {appliance} ({plot_name})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plot_filepath = os.path.join(
        args.save_dir, appliance, f'{plot_name}_loss')
    logger.log(f'Plot directory: {plot_filepath}')
    plt.savefig(fname=plot_filepath)
    if plot_display:
        plt.show()
    plt.close()
    # Mean Absolute Error.
    val_mae = data['val_mae']
    plt.plot(plot_epochs, smooth_curve(val_mae))
    plt.title(f'Smoothed validation MAE for {appliance} ({plot_name})')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plot_filepath = os.path.join(
        args.save_dir, appliance, f'{plot_name}_mae')
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
        default=1024,
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
    parser.set_defaults(do_not_use_distributed_training=False)
    parser.set_defaults(resume_training=False)
    return parser.parse_args()

class TransformerCustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate scheduler per Attention Is All You Need"""
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model_f = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model_f) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps}
        return config

if __name__ == '__main__':
    args = get_arguments()
    appliance_name = args.appliance_name
    logger = Logger(os.path.join(
        args.save_dir,
        appliance_name,
        f'{appliance_name}_train_{args.model_arch}.log')
    )
    logger.log(f'Machine name: {socket.gethostname()}')
    logger.log('Arguments: ')
    logger.log(args)

    model_arch = args.model_arch
    if model_arch not in dir(define_models):
        raise ValueError(f'Unknown model architecture: {model_arch}!')
    else:
        logger.log(f'Using model architecture: {model_arch}.')

    # The appliance to train on.
    appliance_name = args.appliance_name
    logger.log(f'Appliance name: {appliance_name}')

    window_length = common.params_appliance[appliance_name]['window_length']
    logger.log(f'Window length: {window_length}')

    # Path for training data.
    training_path = os.path.join(
        args.datadir,appliance_name,f'{appliance_name}_training_.csv'
    )
    logger.log(f'Training dataset: {training_path}')

    # Look for the validation set
    for filename in os.listdir(os.path.join(args.datadir, appliance_name)):
        if 'validation' in filename:
            val_filename = filename
    # path for validation data
    validation_path = os.path.join(args.datadir,appliance_name, val_filename)
    logger.log(f'Validation dataset: {validation_path}')

    model_filepath = os.path.join(args.save_dir, appliance_name)
    checkpoint_filepath = os.path.join(model_filepath, f'checkpoints_{model_arch}')
    logger.log(f'Checkpoint file path: {checkpoint_filepath}')
    savemodel_filepath = os.path.join(model_filepath, f'savemodel_{model_arch}')
    logger.log(f'SaveModel file path: {savemodel_filepath}')

    # Load datasets.
    train_dataset = common.load_dataset(training_path, args.crop_train_dataset)
    val_dataset = common.load_dataset(validation_path, args.crop_val_dataset)
    NUM_TRAIN_SAMPLES = train_dataset[0].size
    logger.log(f'There are {NUM_TRAIN_SAMPLES/10**6:.3f}M training samples.')
    NUM_VAL_SAMPLES = val_dataset[0].size
    logger.log(f'There are {NUM_VAL_SAMPLES/10**6:.3f}M validation samples.')

    batch_size = args.batchsize
    logger.log(f'Batch size: {batch_size}')

    # Just use gpu 0, mainly for debugging.
    if args.do_not_use_distributed_training:
        strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
    else: # Use all visible GPUs for training.
        strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync
    logger.log(f'Number of replicas: {num_replicas}.')
    global_batch_size = batch_size * num_replicas
    logger.log(f'Global batch size: {global_batch_size}.')

    # Init window generator to provide samples and targets.
    WindowGenerator = common.get_window_generator()
    training_provider = WindowGenerator(
        dataset=train_dataset,
        batch_size=global_batch_size,
        window_length=window_length,
        p=None)# if MODEL_ARCH!='transformer' else 0.2)
    validation_provider = WindowGenerator(
        dataset=val_dataset,
        batch_size=global_batch_size,
        window_length=window_length,
        shuffle=False)

    # Convert Keras Sequence datasets into tf.data.Datasets.
    train_data_iter = lambda: (s for s in training_provider)
    train_tf_dataset = tf.data.Dataset.from_generator(
        train_data_iter,
        output_signature=(
            tf.TensorSpec(shape=(None, window_length, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)))
    train_tf_dataset.prefetch(tf.data.AUTOTUNE)
    val_data_iter = lambda: (s for s in validation_provider)
    val_tf_dataset = tf.data.Dataset.from_generator(
        val_data_iter,
        output_signature=(
            tf.TensorSpec(shape=(None, window_length, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)))
    val_tf_dataset.prefetch(tf.data.AUTOTUNE)

    # Distribute datasets to replicas.
    train_dist_dataset = strategy.experimental_distribute_dataset(train_tf_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(val_tf_dataset)

    try:
        lr = LR_BY_BATCH_SIZE[global_batch_size]
        logger.log(f'Learning rate: {lr}')
    except KeyError as e:
        logger.log('Learning rate cannot be determined due to invalid batch size.')
        raise SystemExit(1) from e

    # Calculate normalized threshold for appliance status determination.
    threshold = common.params_appliance[appliance_name]['on_power_threshold']
    max_on_power = common.params_appliance[appliance_name]['max_on_power']
    threshold /= max_on_power
    logger.log(f'Normalized on power threshold: {threshold}')

    # Get L1 loss multiplier.
    c0 = common.params_appliance[appliance_name]['c0']
    logger.log(f'L1 loss multiplier: {c0}')

    with strategy.scope():
        if model_arch == 'transformer':
            MODEL_DEPTH = 256
            model = define_models.transformer(window_length, d_model=MODEL_DEPTH)
            #lr_schedule = TransformerCustomSchedule(d_model=model_depth)
            #lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                #boundaries=[100000, 200000],
                #values=[5e-04, 1e-04, .5e-04])
            lr_schedule = lr
        elif model_arch == 'cnn':
            model = define_models.cnn()
            lr_schedule = lr
        elif model_arch == 'fcn':
            model = define_models.fcn(window_length)
            lr_schedule = lr
        elif model_arch == 'resnet':
            model = define_models.resnet(window_length)
            lr_schedule = lr
        else:
            logger.log('Model architecture not found.')
            raise SystemExit(1)
        
        # Define loss objects.
        mse_loss_obj = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        mae_loss_obj = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        bce_loss_obj = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )

        def compute_train_loss(y, y_status, y_pred, y_pred_status, model_losses):
            """Calculate per-sample test loss on a single replica for a batch.

            Returns a scalar loss value scaled by the global batch size.
            """

            # Losses on each replica are calculated per batch so a reduce sum is
            # used here which is then scaled by the global batch size.
            mse_loss = tf.reduce_sum(
                mse_loss_obj(y, y_pred)) * (1. / global_batch_size)
            bce_loss = tf.reduce_sum(
                bce_loss_obj(y_status, y_pred_status)
            ) * (1. / global_batch_size)

            # Add mae loss if appliance is on or status is incorrect.
            mask = (y_status > 0) | (y_status != y_pred_status)
            if tf.reduce_any(mask):
                # Apply mask which will flatten tensor. Because of that add
                # a dummy axis so that mae loss is calculated for each batch.
                y = tf.reshape(y[mask], shape=[-1, 1])
                y_pred = tf.reshape(y_pred[mask], shape=[-1, 1])
                # Expected shape = (mask_size, 1)
                mask_size = tf.math.count_nonzero(mask, dtype=tf.float32)
                scale_factor = 1. / (mask_size * (1. * num_replicas))
                mae_loss = c0 * tf.reduce_sum(mae_loss_obj(y, y_pred)) * scale_factor
                loss = mse_loss + bce_loss + mae_loss
            else:
                loss = mse_loss + bce_loss

            # Add model regularization losses.
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))

            return loss

        def compute_test_loss(y, y_status, y_pred, y_pred_status):
            """Calculate per-sample test loss on a single replica for a batch.

            Returns a scalar loss value scaled by the global batch size. This is
            identical to compute_test_loss except model losses are ignored.

            TODO: combine into one loss function with compute_test_loss.
            """

            # Losses on each replica are calculated per batch so a reduce sum is
            # used here which is then scaled by the global batch size.
            mse_loss = tf.reduce_sum(
                mse_loss_obj(y, y_pred)) * (1. / global_batch_size)
            bce_loss = tf.reduce_sum(
                bce_loss_obj(y_status, y_pred_status)
            ) * (1. / global_batch_size)

            # Add mae loss if appliance is on or status is incorrect.
            mask = (y_status > 0) | (y_status != y_pred_status)
            if tf.reduce_any(mask):
                # Apply mask which will flatten tensor. Because of that add
                # a dummy axis so that mae loss is calculated for each batch.
                y = tf.reshape(y[mask], shape=[-1, 1])
                y_pred = tf.reshape(y_pred[mask], shape=[-1, 1])
                # Expected shape = (mask_size, 1)
                mask_size = tf.math.count_nonzero(mask, dtype=tf.float32)
                scale_factor = 1. / (mask_size * (1. * num_replicas))
                mae_loss = c0 * tf.reduce_sum(mae_loss_obj(y, y_pred)) * scale_factor
                loss = mse_loss + bce_loss + mae_loss
            else:
                loss = mse_loss + bce_loss

            return loss

        # Define metrics.
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_mae = tf.keras.metrics.MeanAbsoluteError(name='test_mae')
        test_mse = tf.keras.metrics.MeanSquaredError(name='test_mse')
        mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
        mse = tf.keras.metrics.MeanSquaredError(name='train_mse')

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08
        )

        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
            best_test_loss=tf.Variable(0.0) # MirroredVariable
        )

        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=checkpoint_filepath,
            max_to_keep=3
        )

    # Resume training from last checkpoint or train from scratch.
    # The best test loss is tracked across checkpoints because it is used to
    # determine when to save the best model based on comparing best test loss
    # to current test loss summed from the replicas.
    if args.resume_training:
        if checkpoint_manager.latest_checkpoint is None:
            raise FileNotFoundError('Resume training specified but no checkpoints found.')
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        # best_test_loss is a MirroredVariable that is identical across replicas, so
        # mean reduce it and convert to float for downstream processing.
        best_test_loss = strategy.reduce(
            'MEAN', checkpoint.best_test_loss, axis=None
        ).numpy()
        logger.log(f'Restored checkpoint {checkpoint.save_counter.numpy()}.')
        logger.log(f'Resuming training with best_test_loss {best_test_loss}.')
    else:
        best_test_loss = float('inf')
        logger.log('Training model from scratch.')

    def train_step(data):
        """Runs forward and backward passes on a batch."""
        x, y, y_status = data
        # x expected input shape = (batch_size, window_length)
        # y expected input shape = (batch_size,)
        # status expected input shape = (batch_size,)

        # Include a dummy axis so that reductions are done correctly.
        y = tf.reshape(y, shape=[-1, 1])
        y_status = tf.reshape(y_status, shape=[-1, 1])
        # Expected output shape = (batch_size, 1)

        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            y_pred_status = tf.where(y_pred >= threshold, 1.0, 0.0)
            loss = compute_train_loss(
                y, y_status, y_pred, y_pred_status, model.losses
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        mse.update_state(y, y_pred)
        mae.update_state(y, y_pred)

        return loss

    def test_step(data):
        """Runs forward pass on a batch."""
        x, y, y_status = data
        # x expected input shape = (batch_size, window_length)
        # y expected input shape = (batch_size,)
        # status expected input shape = (batch_size,)

        # Include a dummy axis so that reductions are done correctly.
        y = tf.reshape(y, shape=[-1, 1])
        y_status = tf.reshape(y_status, shape=[-1, 1])
        # Expected output shape = (batch_size, 1)

        y_pred = model(x, training=False)
        y_pred_status = tf.where(y_pred >= threshold, 1.0, 0.0)
        loss = compute_test_loss(y, y_status, y_pred, y_pred_status)

        test_mse.update_state(y, y_pred)
        test_mae.update_state(y, y_pred)

        return loss

    @tf.function
    def distributed_train_step(dataset_inputs):
        """Replicate the train step and run it with distributed input.

        Returns the sum of losses from replicas which is further summed
        over batches and averaged over an epoch in downstream processing.
        """
        per_replica_losses = strategy.run(
            train_step,
            args=(dataset_inputs,)
        )
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_replica_losses,
            axis=None
        )

    @tf.function
    def distributed_test_step(dataset_inputs):
        """Replicate the test step and run it with distributed input.

        Returns the sum of losses from replicas which is further summed
        over batches and averaged over an epoch in downstream processing.
        """
        per_replica_losses =  strategy.run(
            test_step,
            args=(dataset_inputs,)
        )
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_replica_losses,
            axis=None
        )

    wait_for_better_loss = 0
    history = {
        'loss': [],
        'mse': [],
        'mae': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': []
    }

    for epoch in range(args.n_epoch):
        # Train loop.
        total_loss = 0.0
        steps = 0
        pbar = tf.keras.utils.Progbar(
            target=training_provider.__len__(), # number of batches
            stateful_metrics=['mse', 'mae'])
        for d in train_dist_dataset: # iterate over batches
            total_loss += distributed_train_step(d)
            steps += 1 # each step is a batch
            metrics = {
                'loss': total_loss / steps,
                'mse': mse.result(),
                'mae': mae.result()
            }
            pbar.update(
                steps,
                values=metrics.items(),
                finalize=False
            )
        train_loss = total_loss / steps
        pbar.update(steps, values=metrics.items(), finalize=True)

        # Test loop.
        total_loss = 0.0
        steps = 0
        for d in test_dist_dataset:
            total_loss += distributed_test_step(d)
            steps += 1
        test_loss = total_loss / steps

        logger.log(
            f'epoch: {epoch + 1} loss: {train_loss:2.4f} '
            f'mse: {mse.result():2.4f} mae: {mae.result():2.4f} '
            f'val loss: {test_loss:2.4f} '
            f'val mse: {test_mse.result():2.4f} val mae: {test_mae.result():2.4f}'
        )

        if test_loss < best_test_loss:
            logger.log(
                f'Current val loss of {test_loss:2.4f} < '
                f'than val loss of {best_test_loss:2.4f}, '
                f'saving model to {savemodel_filepath}.'
            )
            model.save(savemodel_filepath)
            best_test_loss = test_loss
            checkpoint.best_test_loss.assign(best_test_loss)
            wait_for_better_loss = 0
        else:
            wait_for_better_loss += 1

        checkpoint_manager.save()

        history['loss'].append(train_loss.numpy())
        history['mse'].append(mse.result().numpy())
        history['mae'].append(mae.result().numpy())
        history['val_loss'].append(test_loss.numpy())
        history['val_mse'].append(test_mse.result().numpy())
        history['val_mae'].append(test_mae.result().numpy())

        test_mse.reset_states()
        test_mae.reset_states()
        mse.reset_states()
        mae.reset_states()

        PATIENCE = 6
        if wait_for_better_loss > PATIENCE:
            logger.log('Early termination of training.')
            break

    model.summary()

    plot(history,
         plot_name=f'train_{model_arch}',
         plot_display=True,
         appliance=appliance_name)

"""Class for distributed GPU training.

Copyright (c) 2023~2024 Lindo St. Angel
"""

import tensorflow as tf

from window_generator import WindowGenerator
from numpy import save

class DistributedTrainer():
    """Distributed GPU custom training and test loops.

    Methods:
        train: Training loop which also initiates a test loop after
            one or more epochs are completed.
        train_step: Runs forward and backward passes on a batch.
        distributed_train_step: Replicate the train step and run it with
            distributed input.
        test_step: Runs forward pass on a batch.
        distributed_test_step: Replicate the test step and run it with
            distributed input.

    Attributes:
        num_batches: Number of batches in training dataset.
        num_replicas: Number of distributed replicas in use.
        global_batch_size: Total batch size across replicas.
        strategy: Distribution strategy in use.
        best_test_loss: Best (lowest) loss from last test loop.
    """

    # TODO: fix
    # BASE_LR is a good learning rate for a batch size of 1024
    # which typically leads to best test accuracy. Other batch
    # sizes and learning rates have not been optimized and should
    # be avoided for now.
    _BASE_LR = 1.0e-4
    _LR_BY_BATCH_SIZE = {
        256: _BASE_LR / tf.sqrt(4.0),
        512: _BASE_LR / tf.sqrt(2.0),
        1024: _BASE_LR,
        2048: _BASE_LR * tf.sqrt(2.0),
        4096: _BASE_LR * tf.sqrt(4.0)
    }

    # In single GPU training use this GPU ID.
    _SINGLE_TRAINING_GPU_ID = '/gpu:0'

    def __init__(
            self,
            do_not_use_distributed_training:bool,
            resume_training:bool,
            train_dataset,
            val_dataset,
            batch_size:int,
            model_fn,
            window_length:int,
            checkpoint_filepath:str,
            logger,
            use_mixed_precision:bool=False,
            run_eagerly:bool=False
        ) -> None:
        """Initializes Trainer.

        Args:
            do_not_use_distributed_training: True disables mirrored strategy.
            train_dataset: Training set, tuple of (samples, targets, status).
            val_dataset: Validation set, tuple of (samples, targets, status).
            batch_size: Per replica training batch size.
            window_length: Input sequence window size.
            model_fn: Function to be called that creates model.
            resume_training: True restarts training from last checkpoint.
            checkpoint_filepath: Filepath to checkpoints.
            logger: Logging object.
            use_mixed_precision: True uses `mixed_float16` instead of `TensorFloat-32`.
            run_eagerly: True sets TF eager mode execution.

        Raises:
            SystemExit if unknown model architecture or invalid batch size is specified.
        """
        self._window_length = window_length
        self._logger = logger
        self._wait_for_better_loss = 0 # keeps track of val runs for patience limit

        ### DO NOT USE MIXED-PRECISION - CURRENTLY GIVES POOR MODEL ACCURACY ###
        # TODO: fix
        # Run in mixed-precision mode for ~30% speedup vs TensorFloat-32
        # w/GPU compute capability = 8.6.
        if use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        # Set to True run in TF eager mode for debugging.
        # May have to reduce batch size <= 512 to avoid OOM.
        # Turn off distributed training for best results.
        tf.config.run_functions_eagerly(run_eagerly=run_eagerly)

        # Set distribution strategy and calculate global batch size.
        if do_not_use_distributed_training: # use single gpu
            self._strategy = tf.distribute.OneDeviceStrategy(
                device=self._SINGLE_TRAINING_GPU_ID
            )
        else: # Use all visible GPUs for training.
            self._strategy = tf.distribute.MirroredStrategy()
        self._num_replicas = self._strategy.num_replicas_in_sync
        self._logger.log(f'Number of replicas: {self._num_replicas}.')
        self._global_batch_size = batch_size * self._num_replicas
        self._logger.log(f'Global batch size: {self._global_batch_size}.')

        self._training_provider = WindowGenerator(
            dataset=train_dataset,
            batch_size=self._global_batch_size,
            window_length=self._window_length,
            p=None)# if MODEL_ARCH!='transformer' else 0.2)

        # Create initial training set. This will get reshuffled every epoch.
        self._build_train_dataset()

        self._validation_provider = WindowGenerator(
            dataset=val_dataset,
            batch_size=self._global_batch_size,
            window_length=self._window_length,
            shuffle=False
        )

        # Create validation dataset.
        self._build_val_dataset()

        # Determine learning rate based on global batch size.
        try:
            lr = self._LR_BY_BATCH_SIZE[self._global_batch_size]
            self._logger.log(f'Learning rate: {lr}')
        except KeyError as e:
            self._logger.log('Learning rate cannot be determined due to invalid batch size.')
            raise SystemExit(1) from e

        # Define objects that are distributed across replicas.
        with self._strategy.scope():
            self._model = model_fn()

            # Define loss objects.
            self._mse_loss_obj = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE
            )
            self._mae_loss_obj = tf.keras.losses.MeanAbsoluteError(
                reduction=tf.keras.losses.Reduction.NONE
            )
            self._bce_loss_obj = tf.keras.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            )

            # Define metrics.
            self._test_loss = tf.keras.metrics.Mean(name='test_loss')
            self._test_mae = tf.keras.metrics.MeanAbsoluteError(name='test_mae')
            self._test_mse = tf.keras.metrics.MeanSquaredError(name='test_mse')
            self._mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
            self._mse = tf.keras.metrics.MeanSquaredError(name='train_mse')

            self._optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08
            )

            self._checkpoint = tf.train.Checkpoint(
                optimizer=self._optimizer,
                model=self._model,
                best_test_loss=tf.Variable(0.0) # MirroredVariable
            )

            self._checkpoint_manager = tf.train.CheckpointManager(
                self._checkpoint,
                directory=checkpoint_filepath,
                max_to_keep=2
            )

            self._compute_train_loss_distributed = self._compute_train_loss
            self._compute_test_loss_distributed = self._compute_test_loss

        self._train_from_scratch_or_resume(resume_training)

    def _build_train_dataset(self):
        """Build replica dataset for training."""
        def gen():
            for i in range(len(self._training_provider)): #pylint: disable=consider-using-enumerate
                yield self._training_provider[i]

        train_tf_dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None, self._window_length, 1), dtype=tf.float32), # samples
                tf.TensorSpec(shape=(None,), dtype=tf.float32), # targets
                tf.TensorSpec(shape=(None,), dtype=tf.float32) # status
            )
        )

        # Distribute datasets to replicas.
        self._train_dist_dataset = self._strategy.experimental_distribute_dataset(train_tf_dataset)

    def _build_val_dataset(self):
        """Build replica dataset for testing (validation)."""
        def gen():
            for i in range(len(self._validation_provider)): #pylint: disable=consider-using-enumerate
                yield self._validation_provider[i]

        val_tf_dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None, self._window_length, 1), dtype=tf.float32), # samples
                tf.TensorSpec(shape=(None,), dtype=tf.float32), # targets
                tf.TensorSpec(shape=(None,), dtype=tf.float32) # status
            )
        )

        # Distribute datasets to replicas.
        self._test_dist_dataset = self._strategy.experimental_distribute_dataset(val_tf_dataset)

    def _train_from_scratch_or_resume(self, resume_training):
        """Resume training from last checkpoint or train from scratch.

        best_test_loss is tracked across checkpoints because it is used to
        determine when to save the best model based on comparing best test loss
        to current test loss summed from the replicas.
        """
        if resume_training:
            if self._checkpoint_manager.latest_checkpoint is None:
                raise FileNotFoundError('Resume training specified but no checkpoints found.')
            # Not using assert_consumed() method in restore,
            # see https://github.com/tensorflow/tensorflow/issues/52346
            self._checkpoint.restore(self._checkpoint_manager.latest_checkpoint)
            # best_test_loss is a MirroredVariable that is identical across replicas, so
            # mean reduce it and convert to float for downstream processing.
            self._best_test_loss = self._strategy.reduce(
                'MEAN', self._checkpoint.best_test_loss, axis=None #pylint: disable=no-member
            ).numpy()
            self._logger.log(f'Restored checkpoint: {self._checkpoint.save_counter.numpy()}') #pylint: disable=no-member
            self._logger.log(f'Resuming training with best_test_loss: {self._best_test_loss:.5g}')
        else:
            self._best_test_loss = float('inf')

    def _compute_train_loss(self, y, y_status, y_pred, y_pred_status, model_losses, c0):
        """Calculate per-sample train loss on a single replica for a batch.

        Returns a scalar loss value scaled by the global batch size.
        """

        # Losses on each replica are calculated per batch so a reduce sum is
        # used here which is then scaled by the global batch size.
        mse_loss = tf.reduce_sum(
            self._mse_loss_obj(y, y_pred)
        ) * (1. / self._global_batch_size)
        bce_loss = tf.reduce_sum(
            self._bce_loss_obj(y_status, y_pred_status)
        ) * (1. / self._global_batch_size)

        # Add mae loss if appliance is on or status is incorrect.
        mask = (y_status > 0) | (y_status != y_pred_status)
        if tf.reduce_any(mask):
            # Apply mask which will flatten tensor. Because of that add
            # a dummy axis so that mae loss is calculated for each batch.
            y = tf.reshape(y[mask], shape=[-1, 1])
            y_pred = tf.reshape(y_pred[mask], shape=[-1, 1])
            # Expected shape = (mask_size, 1)
            mask_size = tf.math.count_nonzero(mask, dtype=tf.float32)
            scale_factor = 1. / (mask_size * (1. * self._num_replicas))
            mae_loss = c0 * tf.reduce_sum(self._mae_loss_obj(y, y_pred)) * scale_factor
            loss = mse_loss + bce_loss + mae_loss
        else:
            loss = mse_loss + bce_loss

        # Add model regularization losses.
        if model_losses:
            loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))

        return loss

    def _compute_test_loss(self, y, y_status, y_pred, y_pred_status, c0):
        """Calculate per-sample test loss on a single replica for a batch.

        Returns a scalar loss value scaled by the global batch size.
        """

        # Losses on each replica are calculated per batch so a reduce sum is
        # used here which is then scaled by the global batch size.
        mse_loss = tf.reduce_sum(
            self._mse_loss_obj(y, y_pred)
        ) * (1. / self._global_batch_size)
        bce_loss = tf.reduce_sum(
            self._bce_loss_obj(y_status, y_pred_status)
        ) * (1. / self._global_batch_size)

        # Add mae loss if appliance is on or status is incorrect.
        mask = (y_status > 0) | (y_status != y_pred_status)
        if tf.reduce_any(mask):
            # Apply mask which will flatten tensor. Because of that add
            # a dummy axis so that mae loss is calculated for each batch.
            y = tf.reshape(y[mask], shape=[-1, 1])
            y_pred = tf.reshape(y_pred[mask], shape=[-1, 1])
            # Expected shape = (mask_size, 1)
            mask_size = tf.math.count_nonzero(mask, dtype=tf.float32)
            scale_factor = 1. / (mask_size * (1. * self._num_replicas))
            mae_loss = c0 * tf.reduce_sum(self._mae_loss_obj(y, y_pred)) * scale_factor
            return mse_loss + bce_loss + mae_loss

        return mse_loss + bce_loss

    def train_step(self, data, threshold, c0):
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
            y_pred = self._model(x, training=True)
            y_pred_status = tf.where(y_pred >= threshold, 1.0, 0.0)
            loss = self._compute_train_loss_distributed(
                y, y_status, y_pred, y_pred_status, self._model.losses, c0
            )

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        self._mse.update_state(y, y_pred) #pylint: disable=not-callable
        self._mae.update_state(y, y_pred) #pylint: disable=not-callable

        return loss

    def test_step(self, data, threshold, c0):
        """Runs forward pass on a batch."""
        x, y, y_status = data
        # x expected input shape = (batch_size, window_length)
        # y expected input shape = (batch_size,)
        # status expected input shape = (batch_size,)

        # Include a dummy axis so that reductions are done correctly.
        y = tf.reshape(y, shape=[-1, 1])
        y_status = tf.reshape(y_status, shape=[-1, 1])
        # Expected output shape = (batch_size, 1)

        y_pred = self._model(x, training=False)
        y_pred_status = tf.where(y_pred >= threshold, 1.0, 0.0)
        loss = self._compute_test_loss_distributed(
            y, y_status, y_pred, y_pred_status, c0
        )

        self._test_mse.update_state(y, y_pred) #pylint: disable=not-callable
        self._test_mae.update_state(y, y_pred) #pylint: disable=not-callable

        return loss

    @tf.function
    def distributed_train_step(self, dataset_inputs, threshold, c0):
        """Replicate the train step and run it with distributed input.

        Returns the sum of losses from replicas which is further summed
        over batches and averaged over an epoch in downstream processing.
        """
        per_replica_losses = self._strategy.run(
            self.train_step,
            args=(dataset_inputs, threshold, c0)
        )
        return self._strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_replica_losses,
            axis=None
        )

    @tf.function
    def distributed_test_step(self, dataset_inputs, threshold, c0):
        """Replicate the test step and run it with distributed input.

        Returns the sum of losses from replicas which is further summed
        over batches and averaged over an epoch in downstream processing.
        """
        per_replica_losses = self._strategy.run(
            self.test_step,
            args=(dataset_inputs, threshold, c0)
        )
        return self._strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_replica_losses,
            axis=None
        )

    def _test_loop(self, threshold:float, c0:float, savemodel_filepath:str):
        """Test (validation) loop."""
        total_loss = 0.0
        step = 0
        for d in self._test_dist_dataset:
            total_loss += self.distributed_test_step(d, threshold, c0)
            step += 1
        test_loss = total_loss / step

        if test_loss < self._best_test_loss:
            self._logger.log(
                f'Current val loss of {test_loss:.4g} < '
                f'than val loss of {self._best_test_loss:.4g}, '
                f'saving model to {savemodel_filepath}.'
            )
            self._model.save(savemodel_filepath)
            self._best_test_loss = test_loss #pylint: disable=attribute-defined-outside-init
            self._checkpoint.best_test_loss.assign(self._best_test_loss) #pylint: disable=no-member
            self._wait_for_better_loss = 0
        else:
            self._wait_for_better_loss += 1

        return test_loss

    def train(
            self,
            epochs:int,
            threshold:float,
            c0:float,
            savemodel_filepath:str,
            history_filepath:str,
            patience:int=6,
            test_every_n_epochs:int=1
        ):
        """Training loop.

        This is the main training loop which also initiates one or more test loops
        during an epoch or after one or more epochs are completed.

        Args:
            epochs: Number of epochs to train over.
            threshold: Normalized appliance active threshold.
            c0: Loss function L1 multiplication factor.
            savemodel_filepath: Path to save trained model.
            history_filepath: Path to save model training history as numpy array.
            patience: Number of validation runs to wait for better (lower) loss.
            test_every_n_epoch: Number of epochs to run a validation loop.
        """
        # Structure to save training history.
        history = {
            'loss': [],
            'mse': [],
            'mae': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': []
        }

        # Keeps track of total number of training steps over all batches and epochs.
        total_steps = 0

        for epoch in range(epochs):
            # Train loop.
            total_loss = 0.0
            step = 0
            pbar = tf.keras.utils.Progbar(
                target=len(self._training_provider),
                stateful_metrics=['mse', 'mae']
            )
            for d in self._train_dist_dataset: # iterate over replica batches
                total_loss += self.distributed_train_step(d, threshold, c0)
                step += 1 # each step is a replica batch
                metrics = {
                    'loss': total_loss / step,
                    'mse': self._mse.result(), #pylint: disable=not-callable
                    'mae': self._mae.result() #pylint: disable=not-callable
                }
                pbar.update(
                    step,
                    values=metrics.items(),
                    finalize=False
                )
                # Update training history.
                for k, v in metrics.items():
                    history[k].append((step + total_steps, v.numpy()))
            pbar.update(step, values=metrics.items(), finalize=True)

            # Run test loop at end of every one (n=1) or every n (n>1) epochs.
            if epoch % test_every_n_epochs == 0:
                self._logger.log(f'Running test loop after epoch: {epoch + 1}.')
                test_loss = self._test_loop(threshold, c0, savemodel_filepath)
                # Update test history.
                history['val_loss'].append((step + total_steps, test_loss.numpy()))
                history['val_mse'].append((step + total_steps, self._test_mse.result().numpy())) #pylint: disable=not-callable
                history['val_mae'].append((step + total_steps, self._test_mae.result().numpy())) #pylint: disable=not-callable

            total_steps += step

            # Log all metrics at end of current epoch.
            # TODO: fix
            #   Sadly can't use f-strings here, get
            #   `TypeError: unsupported format string passed to tuple.__format__`
            s = (
                'epoch: {} '
                'loss: {:.4g} mse: {:.4g} mae: {:.4g} '
                'val_loss: {:.4g} val_mse: {:.4g} val_mae: {:.4g}'
            )
            self._logger.log(
                s.format(
                    epoch + 1,
                    history['loss'][-1][-1], history['mse'][-1][-1], history['mae'][-1][-1],
                    history['val_loss'][-1][-1], history['val_mse'][-1][-1], history['val_mae'][-1][-1]
                )
            )

            # Save checkpoint at end of current epoch.
            self._checkpoint_manager.save()

            # Save history.
            save(history_filepath, history)

            if self._wait_for_better_loss == patience:
                self._logger.log('Early termination of training.')
                break

            # Reset metrics.
            self._test_mse.reset_states()
            self._test_mae.reset_states()
            self._mse.reset_states()
            self._mae.reset_states()

            # Reshuffle training dataset.
            self._logger.log('Reshuffling training dataset.')
            self._training_provider.on_epoch_end()
            self._build_train_dataset()

        self._model.summary()

        return history

    @property
    def num_batches(self):
        """Returns number of batches in training dataset."""
        return len(self._training_provider)

    @property
    def num_replicas(self):
        """Returns number of distributed replicas in use."""
        return self._num_replicas

    @property
    def global_batch_size(self):
        """Returns total batch size across replicas."""
        return self._global_batch_size

    @property
    def strategy(self):
        """Returns distribution strategy in use."""
        return self._strategy

    @property
    def best_test_loss(self):
        """Returns best (lowest) loss from last test loop."""
        return self._best_test_loss

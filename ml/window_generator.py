"""Class for Window generator.

Copyright (c) 2023 Lindo St. Angel
"""

import numpy as np

class WindowGenerator():
    """Generates windowed time series samples, targets and status.

    If 'p' is not None the input samples are processed with random masking, 
    where a proportion 'p' of input elements are randomly masked with a 
    special token and only output results from such positions are used to 
    compute the loss using a keras model fit() custom train step. This may be 
    useful in training transformers in a masked language model fashion (MLM). See:
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    (https://arxiv.org/pdf/1810.04805.pdf).

    Methods:
        on_epoch_end(): Shuffle datasets, usually called at the end of an training epoch.
        __getitem(index)__: Special method that is used to access batches of windowed data
            items by an index.
        __len()__: Special method that returns the number of windowed data items which
            is equal to the number of batches in an epoch.
    
    Attributes:
        None
    """
    # The mask value used in masked language model training.
    _MASK_TOKEN = -1.0

    def __init__(
        self,
        dataset,
        batch_size=1024,
        window_length=599,
        train=True,
        shuffle=True,
        allow_partial_batches=False,
        p=None
    ) -> None:
        """Inits WindowGenerator.

        Args:
            dataset: Tuple of input time series samples, targets, activations.
            batch_size: Batch size.
            window_length: Number of samples in a window of time series data.
            train: True returns samples and targets else just samples.
            shuffle: True shuffles dataset initially and when on_epoch_end is called.
            allow_partial_batches: True allows partial batches per epoch.
            p: Proportion of input samples masked with a special token.
        """
        x, y, activations = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._window_length = window_length
        self._train = train

        # Calculate window center index.
        self._window_center = int(0.5 * (self._window_length - 1))

        # Number of input samples adjusted for windowing.
        # This prevents partial window generation.
        self._num_samples = x.size - self._window_length

        # Calculate number of batches in an epoch.
        if allow_partial_batches:
            self._batches_per_epoch = int(np.ceil(self._num_samples / self._batch_size))
        else:
            self._batches_per_epoch = self._num_samples // self._batch_size

        # Generate indices of adjusted input sample array.
        self._indices = np.arange(self._num_samples)

        self._rng = np.random.default_rng()

        self._samples, self._targets, self._status = self._randomly_mask_input(
            x.size, x, y, activations, p
        ) if p else x, y, activations

        # Initial shuffle.
        if self._shuffle:
            self._rng.shuffle(self._indices)

    def _randomly_mask_input(self, num_samples, x, y, activations, p):
        """Randomly mask input sequence."""
        samples, targets, status = [], [], []
        for i in range(num_samples):
            prob = self._rng.random()
            if prob < p:
                prob = self._rng.random()
                if prob < 0.8:
                    samples.append(self._MASK_TOKEN)
                elif prob < 0.9:
                    samples.append(self._rng.normal())
                else:
                    samples.append(x[i])
                targets.append(y[i])
                status.append(activations[i])
            else:
                samples.append(x[i])
                targets.append(self._MASK_TOKEN)
                status.append(self._MASK_TOKEN)
        return samples, targets, status

    def on_epoch_end(self) -> None:
        """Shuffle at end of each epoch.""" 
        if self._shuffle:
            self._rng.shuffle(self._indices)

    def __len__(self) -> int:
        """Returns number batches in an epoch."""
        return self._batches_per_epoch

    def __getitem__(self, index):
        """Returns batches of windowed samples, targets and status."""
        # Row indices for current batch.
        rows = self._indices[
            index * self._batch_size:(index + 1) * self._batch_size
        ]

        # Create a batch of windowed samples.
        windowed_samples = np.array(
            [self._samples[row:row + self._window_length] for row in rows]
        )

        # Add 'channel' axis for model input.
        windowed_samples = windowed_samples.reshape((-1, self._window_length, 1))

        if self._train:
            # Create batch of window-centered, single point targets and status.
            windowed_targets = np.array(
                [self._targets[row + self._window_center] for row in rows]
            )
            windowed_status = np.array(
                [self._status[row + self._window_center] for row in rows]
            )

            return windowed_samples, windowed_targets, windowed_status

        # Return only samples if in test mode.
        return windowed_samples

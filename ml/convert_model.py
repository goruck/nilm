"""Class for converting quantizing Keras models to tflite.

Copyright (c) 2023 Lindo St. Angel
"""

import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ConvertModel():
    """Convert a Keras model to tflite and quantize.

    The conversion optimizes for latency and size and various quantization schemes
    are supported including quantizing weights and activations as well as input and
    output layers. Public methods are provided for debugging the quantized model and
    keeping troublesome layers and ops in float to improve accuracy.

    Attributes:
        None.
    """

    def __init__(
            self,
            keras_model,
            quant_mode,
            sample_provider,
            num_cal,
            log,
            debug_results_filepath,
            debug_results_filename,
            debug_results_plot_name
        ) -> None:
        """Initializes instance and configures it based on quantization mode.

        Args:
            keras_model: Keras input model.
            sample_provider: Object that provides samples for calibration.
            log: Logger object.
            quant_mode: Quantization mode.
            num_cal: Number of samples for post-training quantization calibration.
            debug_results_filepath: Path to save the debugger results.
            debug_results_filename: Name of raw debugger results file.
            debug_results_plot_name: Name of debugger result plot file.
        
        Raises:
            ValueError: Unrecognized quantization mode.
        """

        self.sample_provider = sample_provider
        self.num_cal = num_cal
        self.logger = log
        self.quant_mode = quant_mode
        self.debug_results_filepath = debug_results_filepath
        self.debug_results_filename = debug_results_filename
        self.debug_results_plot_name = debug_results_plot_name

        self.converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

        # Converter optimization settings.
        # The DEFAULT optimization strategy quantizes model weights and
        # activations and tries to optimize for size and latency, while
        # minimizing the loss in accuracy.
        # The EXPERIMENTAL_SPARSITY optimization tries to advantage of
        # the sparse model weights trained with pruning if available.
        self.converter.optimizations = [
            tf.lite.Optimize.DEFAULT,
            #tf.lite.Optimize.EXPERIMENTAL_SPARSITY #TODO will not work with debugger
        ]

        if self.quant_mode == 'w8':
            # Quantize only weights from floating point to int8.
            # Inputs and outputs are kept in float.
            self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        elif self.quant_mode == 'w8_a8_fallback':
            # Quantize weights and activations from floating point to int8 with
            # fallback to float if an operator does not have an int implementation.
            # Inputs and outputs are kept in float.
            self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            self.converter.representative_dataset = tf.lite.RepresentativeDataset(self._rep_gen)
        elif self.quant_mode == 'w8_a8':
            # Enforce full int8 quantization for all ops including the input and output.
            self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            self.converter.representative_dataset = tf.lite.RepresentativeDataset(self._rep_gen)
            self.converter.inference_input_type = tf.int8
            self.converter.inference_output_type = tf.int8
        elif self.quant_mode == 'w8_a16':
            # Quantize activations based on their range to int16, weights
            # to int8 and bias to int64 with fallback to float for
            # unsupported operators. Inputs and outputs are kept in float.
            self.converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            self.converter.representative_dataset = tf.lite.RepresentativeDataset(self._rep_gen)
        else:
            raise ValueError('Unrecognized quantization mode.')

    def _rep_gen(self) -> np.float32:
        """Yields samples from representative dataset.

        This is a generator function that provides a small dataset to calibrate or
        estimate the range, i.e, (min, max) of all floating-point arrays in the model.

        Since dataset is very imbalanced must ensure enough samples are used to generate a
        representative set. 96 hours (4 days) of samples is a good starting point as that
        period should be long enough to capture infrequently used appliances.

        Yields:
            A representative model input sample.

        Raises:
            ValueError: Not enough representative samples to run calibration.
            RunTimeError: Bad sample in representative dataset.
        """

        # Get number of samples per batch in provider. Since batch should always be
        # set to 1 for inference, this will simply return the total number of samples.
        samples_per_batch = len(self.sample_provider)

        if self.num_cal > samples_per_batch:
            raise ValueError('Not enough representative samples.')

        # Generate 'num_cal' samples at random locations in dataset.
        #indices = rng.choice(samples_per_batch, size=num_cal)
        #for i in indices:
            #sample, _, _ = sample_provider.__getitem__(i)
            #yield [sample,]

        # Generate 'num_cal' samples from provider.
        # Given the dataset is unbalanced, this logic ensures that sufficient
        # representative samples from active appliances periods are yielded.
        active_frac = 0.5 # fraction of cal samples during appliance active times
        num_cal_active = int(active_frac * self.num_cal)
        i = 0
        active = 0
        inactive = 0
        while True:
            if i > samples_per_batch: # out of samples
                self.logger.log(
                    f'Out of cal samples, act={active}, inact={inactive}.', level='warning'
                )
                break
            sample, _, status = self.sample_provider[i]
            if sample.size == 0 or status.size == 0:
                raise RuntimeError(f'Missing representative data at i={i}.',)
            status = bool(status.item())
            i+=1
            if active < num_cal_active:
                if status:
                    active+=1
                    yield [sample,] # generate samples from active times...
                continue
            if active + inactive < self.num_cal:
                if not status:
                    inactive+=1
                    yield [sample,] # ... then generate samples from inactive times
            else:
                break

    def _get_suspected_layers(
            self, layer_stats:pd.DataFrame,
            high_quant_error:float=0.7,
            remove_large_ranges:bool=False,
            remove_first_five_layers:bool=False
        ) -> list:
        """Generate a list of suspected layers given model debug results.

        The RMSE / scale is close to 1 / sqrt(12) (~ 0.289) when quantized
        distribution is similar to the original float distribution, indicating
        a well quantized model. The larger the value is, it's more likely for
        the layer not being quantized well. These layers can remain in float to
        generate a selectively quantized model that increases accuracy at the
        expense of inference performance. See:
        https://www.tensorflow.org/lite/performance/quantization_debugger
        """
        suspected_layers = list(
            layer_stats[layer_stats['rmse/scale'] > high_quant_error]['tensor_name']
        )

        # Keep layers with range greater than 255 (8-bits) in float as well.
        if remove_large_ranges:
            return suspected_layers.extend(
                list(layer_stats[layer_stats['range'] > 255.0]['tensor_name'])
            )

        # Let the first 5 layers remain in float as well.
        if remove_first_five_layers:
            return suspected_layers.extend(list(layer_stats[:5]['tensor_name']))

        return suspected_layers

    def convert(self):
        """Quantize model.

        Returns:
            Quantized model in tflite format.
        """
        return self.converter.convert()

    def debug(self):
        """Quantize model and check how well it was quantized.

        Works only for full INT8 quantization.

        Returns:
            Quantized model in tflite format.

        Raises:
            ValueError: Full INT8 quantization mode not specified.
        """
        self.logger.log('Debugging model...')

        if self.quant_mode in ['w8', 'w8_a16']:
            raise ValueError('Model debug only works for full INT8 quant.')

        debugger = tf.lite.experimental.QuantizationDebugger(
            converter=self.converter,
            debug_dataset=self._rep_gen
        )
        debugger.run()

        debug_results_file = os.path.join(
            self.debug_results_filepath, self.debug_results_filename
        )
        with open(debug_results_file, mode='w', encoding='utf-8') as f:
            debugger.layer_statistics_dump(f)

        pd.set_option('display.max_rows', None)
        layer_stats = pd.read_csv(debug_results_file)
        self.logger.log(layer_stats)

        layer_stats['range'] = 255.0 * layer_stats['scale']
        layer_stats['rmse/scale'] = layer_stats.apply(
            lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)
        self.logger.log(layer_stats[['op_name', 'range', 'rmse/scale']])

        plt.figure(figsize=(15, 5))
        ax1 = plt.subplot(121)
        ax1.bar(np.arange(len(layer_stats)), layer_stats['range'])
        ax1.set_ylabel('range')
        ax2 = plt.subplot(122)
        ax2.bar(np.arange(len(layer_stats)), layer_stats['rmse/scale'])
        ax2.set_ylabel('rmse/scale')
        debug_results_plot = os.path.join(
            self.debug_results_filepath, self.debug_results_plot_name
        )
        self.logger.log(f'Saving debug results plot to {debug_results_plot}.')
        plt.savefig(fname=debug_results_plot)

        suspected_layers = self._get_suspected_layers(layer_stats)
        self.logger.log(f'Suspected layers: {suspected_layers}')

        return debugger.get_nondebug_quantized_model()

    def fix(self, deny_ops:list=None):
        """Quantize model but keep troublesome layers and ops in float.

        Args:
            deny_ops: List of operators to keep in float.

        Returns:
            Quantized model in tflite format.
        """
        debug_results_file = os.path.join(
            self.debug_results_filepath, self.debug_results_filename
        )
        self.logger.log(f'Fixing model using debug results file: {debug_results_file}.')
        layer_stats = pd.read_csv(debug_results_file)
        layer_stats['range'] = 255.0 * layer_stats['scale']
        layer_stats['rmse/scale'] = layer_stats.apply(
            lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)
        suspected_layers = self._get_suspected_layers(layer_stats)
        self.logger.log(f'Layers kept in float: {suspected_layers}')
        self.logger.log(f'Ops kept in float: {deny_ops}')
        debug_options = tf.lite.experimental.QuantizationDebugOptions(
            denylisted_nodes=suspected_layers,
            denylisted_ops=deny_ops
        )
        debugger = tf.lite.experimental.QuantizationDebugger(
            converter=self.converter,
            debug_dataset=self._rep_gen,
            debug_options=debug_options
        )
        return debugger.get_nondebug_quantized_model()

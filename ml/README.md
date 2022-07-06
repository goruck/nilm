# ml

This folder contains most of the machine learning related code for training and testing the models. See main README for a general overview and the below for a brief description for each script or module.

`cnn_model.py` - Deprecated and replaced by `define_models.py`.

`common.py` - Various common functions and parameters.

`convert_keras_to_tflite.py` - Generates a quantized tflite model that can be complied for the edge tpu
or used as-is for on device inference on a Raspberry Pi and other edge compute.

`define_models.py` - Creates Keras models for seq2point learning.

`logger.py` - Configures Python logging.

`nilm_metric.py` - Computes test statistics.

`predict.py` - Predict appliance type and power with locally captured novel data using trained Keras floating point models and, optionally, overlay Raspberry PI / tflite based inference results.

`quantize.py` - Deprecated and replaced by `convert_keras_to_tflite.py`.

`test.py` - Main script used for training the models from scratch with options to prune and for quantize aware training.

`train.py` - Used for testing the trained models on datasets disjoint from test and validation data.
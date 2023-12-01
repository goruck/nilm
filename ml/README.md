# ml

This folder contains most of the machine learning related code for training and testing the models. See main README for a general overview and the below for a brief description for each item in this folder.

**dataset_management/** - Folder that contains various scripts for generating and visualizing REDD, REFIT, UKDALE and locally captured datasets. The generated data includes training, validation and test sets for training and testing the seq2point deep learning models.

**images/** - Folder that contains various visualizations and test results as image files.

**models/** - Folder that contains the fitted models per appliance with images of their loss curves, training results and test results.

**plots/** - Folder that contains various scripts used in interpreting and visualizing the models. 

`cnn_model.py` - Deprecated and replaced by `define_models.py`.

`common.py` - Module that contains various common functions and parameters.

`convert_keras_to_tflite.py` - Script that generates a quantized tflite model that can be complied for the edge tpu or used as-is for on device inference on a Raspberry Pi and other edge compute.

`define_models.py` - Script that creates Keras models for seq2point learning.

`logger.py` - Script that configures Python logging.

`nilm_metric.py` - Script that computes test statistics.

`predict.py` - Script that predicts appliance type and power with locally captured novel data using trained Keras floating point models and, optionally, overlay Raspberry PI / tflite based inference results.

`quantize.py` - Deprecated and replaced by `convert_keras_to_tflite.py`.

`train_keras.py` - Script used for training the models from scratch with options to prune and for quantize aware training.

`train_distributed.py` - Script used for training the models from scratch or resumption of training using distributed GPU compute. Mainly supersedes `train.py`.

`test.py` - Script used for testing the trained models on data disjoint from test and validation data.

`visualize_keras_model.py` - Script that generates network diagrams from Keras models.

`transformer_model.py` - Defines NILM transformer-based model using Keras subclassed layers and models.

`distributed_trainer.py` - Class for `test.py`,

`window_generator` - Class for window generator used by many modules in this project.

`convert_model.py` - Class for converting quantizing Keras models to tflite. Used by `convert_keras_to_tflite.py`
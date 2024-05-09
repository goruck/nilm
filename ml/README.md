# ml

This folder contains most of the machine learning related code for training and testing the models. See main README for a general overview and the below for a brief description for each item in this folder.

NB: If using TensorFlow 2.16+, then run the scripts in this folder after exporting the environment variable TF_USE_LEGACY_KERAS=1 since TF 2.16+ uses Keras 3 by default and this code was developed using Keras 2. See https://keras.io/keras_3/.

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

`train_keras.py` - Script used for training the models from scratch with options to prune and for quantize aware training. Depreciated by `train.py`.

`train.py` - Script used for training the models from scratch or resumption of training using distributed GPU compute. Supersedes `train_keras.py`.

`fine_tune.py` - Script used for fine tuning models using locally captured data.

`test.py` - Script used for testing the trained models on data disjoint from test and validation data.

`visualize_keras_model.py` - Script that generates network diagrams from Keras models.

`transformer_model.py` - Defines NILM transformer-based model using Keras subclassed layers and models.

`distributed_trainer.py` - Distributed trainer Class used in `train.py` and `fine_tune.py`.

`window_generator` - Class for window generator used by many modules in this project.

`convert_model.py` - Class for converting quantizing Keras models to tflite. Used by `convert_keras_to_tflite.py`
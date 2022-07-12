# tpu

This folder contains the Google Coral Edge TPU related code. See main README for a general overview and the information here for additional details.

The script `ml/convert_keras_to_tflite.py` will generate an intermediate tflite model that is processed with the Edge TPU Compiler (`edgetpu_compiler`) which is a command line tool that compiles this model into a file that's compatible with the Edge TPU. See the [Edge TPU Compiler documentation](https://coral.ai/docs/edgetpu/compiler/) for more information. 

Once the model is compiled, the script `evaluate_tflite_model.py` can be used to evaluate the model on the tpu for the metrics MAE, NDE, SAE and inference rate. It can be used to compare the performance of a model on the tpu vs other machines. Thus far most models used in this project perform better on the Raspberry Pi 4 than on he Edge TPU so no further development will be done on the tpu code at this time. 
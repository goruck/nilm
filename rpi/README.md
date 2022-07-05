# rpi

This folder contains all Raspberry Pi related code. See main README for a general overview and the below for a brief description for the main scripts.

`infer.py` - This runs inference on a per appliance basis, performs pre- and post-processing and stores observations in a csv file for downstream processing. It is meant to be run continuously. 

`evaluate_tflite_model.py` - This evaluates a tflite model on the raspberry pi for the metrics MAE, NDE, SAE and inference rate. It can be used to compare the performance of a model on the rpi vs other machines. 
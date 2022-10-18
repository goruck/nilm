# Energy Management Using Real-Time Non-Intrusive Load Monitoring

## Introduction

The goal of non-intrusive load monitoring (NILM) is to recover the energy consumption of individual appliances from the aggregate mains signal, which is a measure of the total electricity consumption of a building or house. NILM is also known as energy disaggregation and both terms will be used interchangeably throughout.

R. Gopinath, et al., nicely summarizes the rational behind NILM in the paper Energy management using non-intrusive load monitoring techniques — State- of-the-art and future research directions¹:

> In recent years, the development of smart sustainable cities has become the primary focus among urban planners and policy makers to make responsible use of resources, conserve the environment and improve the well-being of the society. Energy management is an integral part of the smart sustainable cities development programme which involves conscious and efficient use of available energy resources towards attaining sustainability and self-reliance on energy systems. Building sector is one of the key sectors that utilize more energy. Therefore, efforts are being made to monitor and manage energy consumption effectively in residential and commercial buildings. In recent years, non-intrusive load monitoring (NILM) technique has become a popular and emerging approach to monitor events (on/off) and energy consumption of appliances/electrical utilities in buildings using single energy meter. The information about the energy consumption at the appliance level would help consumers to understand their appliance usage behavior and take necessary steps for reducing energy consumption.

By using deep learning models trained on publicly available datasets it is now feasible to enable NILM in very cost effective ways by leveraging commodity hardware and open source software. Deploying the trained models at the edge of the power grid, i.e. the building level, results in additional cost savings since it obviates the need for always connected internet cloud services which can be expensive at scale and running the algorithms in real-time provides for low latency operation.

This Project will show you how NILM works by taking you through the steps I used to implement a prototype system at my house.

## Architecture

The following first two diagrams illustrate the NILM concept and process steps at a high level. The last diagram shows my prototype system which is based on these concepts and is implemented using Arduino- and Raspberry Pi-based compute.


![Alt text](./img/nilm-general-concept.png?raw=true "General NILM Concept (R. Gopinath, et al.¹)")

![Alt text](./img/nilm-scheme.png?raw=true "NILM Process Steps (R. Gopinath, et al.¹)")

![Alt text](./img/system-blk-dia.png?raw=true "NILM Prototype System Block Diagram")

## NILM Algorithm Selection and Model Training

### Algorithm Selection

Energy disaggregation is a highly underdetermined and a single-channel Blind Source Separation (BSS) problem² which makes it difficult to obtain accurate predictions. You need to extract more than one source from a single observation. Past approaches have included factorial hidden Markov models (FHMM)¹ and various event-based methods with some success⁶.

You can also solve the single-channel BSS problem can by using sequence-to-sequence (seq2seq) learning with neural networks, and it can applied to the NILM problem using both convolutional and recurrent neural networks³. Seq2seq learning involves training a deep network to map between an input time series, such as the aggregate power readings the case of NILM, and a output sequence, such as the estimated energy consumption of a single appliance. A sliding input window is typically used to training the network which generates a corresponding window of the output. This method produces multiple predictions for each appliance in the output so an average of the predictions is used for the final result. Some of the predictions will be more accurate than others, especially those near the midpoint of the input sliding window. The averaging will tend to lower the overall accuracy of the predictions.

Some of the disadvantages of of seq2seq leaning can mitigated by sequence-to-point learning (seq2point) for single-channel BSS⁴. You also use a sliding input signal window in this approach, however the network is trained to predict the output signal only at the midpoint of the window which makes the prediction problem easier on the network, leading to more accurate results.

I selected the seq2point learning approach for my prototype system and my implementation was inspired and guided by work described by Michele D'Incecco, Stefano Squartini and Mingjun Zhong⁵.

### Datasets

There are a number of large-scale publicly available datasets specifically designed to address the NILM problem which were captured in household buildings from various countries. The table⁷ below shows several of the most widely used.

![Alt text](./img/nilm-datasets.png?raw=true "NILM Datasets (Oliver Parson, et al.⁷)")

The datasets generally include many 10’s of millions of active power, reactive power, current, and voltage samples but with different sampling frequencies which requires you to pre-process the data before use. Most NILM algorithms utilize only real (aka active or true) power data. Five appliances are usually considered for energy disaggregation research which are kettle, microwave, fridge, dish washer and washing machine. These are the appliances I considered for my prototype and following the work of Michele DIncecco, et al.⁵, I mainly focused on the REFIT⁸ data et but will eventually include UK-DALE and REDD.

### Model Training

I used TensorFlow 2 and the Keras APIs to train and test the model. All code associated with this section can be found on the Machine Learning section of the project’s GitHub, NILM⁹.

The seq2point learning models for the appliances were trained individually on z normalized REFIT data. I used the following four metrics to evaluate the model’s performance. You can view the code that calculates these metrics [here](./ml/nilm_metric.py).

* Mean absolute error (MAE), which evaluates the absolute difference between the prediction and the and the ground truth at every time point and calculates the mean value.
* Normalized signal aggregate error (SAE), which indicates the relative error of the total energy.
* Energy per day (EpD) which measures the absolute error of the predicted energy used in a day, useful when the household users are interested in the total energy consumed in a period.
* Normalized disaggregation error (NDE) which measures the normalized error of the squared difference between the prediction and the ground truth of the appliances.

I created the seq2point learning model `create_model` using the Keras Functional API as shown in the Python module [define_models.py](./ml/define_models.py). You can find alternative model architectures there as well but this one currently gives the best results.

A sliding window of 599 samples of the aggregate real power consumption signal is used as inputs to seq2point model and the midpoints of the corresponding windows of the appliances are used as targets. You can see how the samples and targets are generated in the `get_window_generator` function in the [common.py](./ml/common.py) module.

You can see the code I used to train the model in [train.py](./ml/train.py). I used the Keras Adam optimizer and to reduce over-fitting I used EarlyStopping and InverseTimeDecay. The training code can be configured to train the seq2point model from scratch, or given a fitted model, prune it or fine tune it with quantization aware training (QAT), both of which can improve inference performance especially on edge hardware.

The hyper-parameters for training and the optimizer are summarized below.

* Input Window Size: 599 samples
* Batch size: 1000 samples.
* From scratch Learning Rate: 0.001 with inverse time decay.
* QAT and Prune Learning Rate: 0.0001.
* Optimizer: beta_1=0.9, beta_2=0.999, epsilon=1e-08.
* From scratch Early Stopping Criteria: 6 epochs.

The training program monitors the Mean Squared Error (MSE) losses for both training and validation data and Mean Absolute Error (MAE) for the validation data with early stopping to reduce over-fitting. The datasets contain a large number of samples (many 10’s of millions) with repeating patterns; it was not uncommon that over-fitting occurred after only a single epoch for some appliances. To mitigate against this, I used a subset of the training data, typically between 5 and 10 million samples.

You can find the training results for each appliance in the [models](./ml/models/) folder and typical performance metrics in the Appendix.

### Model Quantization

I quantized the model’s weights and activation functions from Float32 to INT8 using TensorFlow Lite to improve inference performance on edge hardware, including the Raspberry Pi and the Google Edge TPU. See [convert_keras_to_tflite.py](./ml/convert_keras_to_tflite.py) for the code that does this quantization. You may observed a slight degradation in performance after quantization but this is acceptable for most use cases. A typical result is shown below, note that the floating point model was not fine-tuned using QAT nor pruned before INT8 conversion.

```text
### Fridge Float32 vs INT8 Model Performance ###

Float32 tflite Model (Running on x86 w/XNNPACK):
MAE 17.10(Watts)
NDE 0.373
SAE 0.283

Inference Rate 701.0 HzINT8 tflite Model (running on Raspberry Pi 4):
MAE: 12.75 (Watts)
NDE 0.461
SAE 0.120
Inference Rate 163.5 Hz
```

## NILM Prototype System Components

I built a NILM prototype at my home to test the energy disaggregation algorithms in real-world conditions and to understand where they can be improved. The prototype is comprised of the following main subsystems. You can see a photograph of the prototype in the Appendix.

### Analog Signal Conditioning

I used two clip-on current transformers in one of the home’s sub-panels to sense the current flowing through each of split voltage phases and a voltage transformer plugged into an outlet near the circuit breaker panel that provides the voltage of one of the phases. These signals are level-shifted, amplified and low-passed filtered by this subsystem before being passed on to the analog-to-digital converters inside an Arduino MEGA 2560 that performs aggregate metrics computation. You can see a schematic for the Analog Signal Conditioning Subsystem in the Appendix and find more details in the [Panel to Arduino section](./pan-ard-inf/README.md).

### Aggregate Metrics Computation

I used an Arduino MEGA 2560 to host the signal processing algorithms that takes the voltage signals from the Analog Signal Conditioning Subsystem and generates aggregate RMS voltage, RMS current, Real Power and Apparent Power metrics in real-time. Presently, only Real Power is used in downstream processing. I leveraged emonLibCM<sup>10</sup> for these signal processing algorithms. emonLibCM runs continuously in the background and digitizes the analog input channels of the Arduino, calculates these metrics and then informs the Arduino sketch that the measurements are available and should be read and processed by downstream processing. The sketch is configured to update the metrics every eight seconds and can be found in [ard.ino](./ard/ard.ino)

### Disaggregated Energy Consumption Computation

The actual energy disaggregation computations are hosted on a Raspberry Pi 4 which is connected over USB to the Arduino to fetch the aggregate metrics. The computations are comprised of running the tflite appliance inference models, trained and quantized per the steps described above, with pre- and post-processing steps. See the [infer.py](./rpi/infer.py) module for the code that performs these computations. The inference models output predicted energy for each appliance from 599-sample sized windows of the aggregate real power input signal. These predictions are stored in a local CSV file and made available for downstream reporting and analysis.

Typical disaggregated energy prediction results from my home are shown in the figures below using tflite models trained on another machine from the dataset as described above.

![Alt text](./img/garage-prediction-results.png?raw=true "Garage Prediction Results")

![Alt text](./img/garage-prediction-zoomed.png?raw=true "Zoomed Garage Prediction Results")

No fine-tuning was done on local data. The horizontal axis is time measured by the number of eight-second samples (the sampling interval is eight seconds), which span about eight and half days in this case. The vertical axis is energy consumption in Watts. The top trace is total (both phases) mains real power. The following traces are the predicted energy for each appliance. Only a refrigerator and a microwave from the reference set of appliances were in use during this time although there were other devices supplied by the same sub-panel such as computers and heaters.

You can see both the Float32 and tflite versions algorithm detect the refrigerator and microwave but there is an energy offset between the two which probably indicates the need for better calibration. I used a power meter to ground truth the RMS power consumption of the refrigerator and microwave; it matches the Float32 predictions to within about +/- 15%.

### Amazon Alexa

I plan to use Amazon Alexa as the user interface to the appliance energy data, however this work is not yet started.

## Conclusion

By using large publicly available datasets to train seq2point learning models it is very feasible to perform energy disaggregation that is fairly accurate without fine-tuning the models with local data. These models are modest in size and with little loss in accuracy can be quantized to run efficiently on commodity edge hardware such as the Raspberry Pi 4. More work needs to be done to further improve the accuracy of the models and to test with more appliance types. This project demonstrates that a key component of a sustainable and scalable power grid is within reach of the mass consumer market.

## References

1. Sustainable Cities and Society 62 (2020) 102411 | Energy management using non-intrusive load monitoring techniques — State- of-the-art and future research directions by R. Gopinath, Mukesh Kumar, C. Prakash Chandra Joshua and Kota Srinivas.
2. Wikipedia | Signal Separation.
3. arXiv:1507.06594 | Neural NILM: Deep Neural Networks Applied to Energy Disaggregation by Jack Kelly and William Knottenbelt.
4. arXiv:1612.09106 | Sequence-to-point learning with neural networks for non-intrusive load monitoring by Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard and Charles Sutton.
5. arXiv:1902.08835 | Transfer Learning for Non-Intrusive Load Monitoring by Michele D'Incecco, Stefano Squartini and Mingjun Zhong.
6. Artificial Intelligence Techniques for a Scalable Energy Transition pp 109–131 | A Review on Non-intrusive Load Monitoring Approaches Based on Machine Learning by Hajer Salem, Moamar Sayed-Mouchaweh and Moncef Tagina.
7. 1st International Symposium on Signal Processing Applications in Smart Buildings at 3rd IEEE Global Conference on Signal & Information Processing | Dataport and NILMTK: A Building Data Set Designed for Non-intrusive Load Monitoring by Oliver Parson, Grant Fisher, April Hersey, Nipun Batra, Jack Kelly, Amarjeet Singh, William Knottenbelt and Alex Rogers.
8. Proceedings of the 8th International Conference on Energy Efficiency in Domestic Appliances and Lighting | A data management platform for personalised real-time energy feedback by David Murray and Jing Liao and Lina Stankovic and Vladimir Stankovic and Richard Hauxwell-Baldwin and Charlie Wilson and Michael Coleman and Tom Kane and Steven Firth. The REFIT dataset used in this project is is licensed under the Creative Commons Attribution 4.0 International Public License.
9. GitHub | NILM by Lindo St. Angel
10. GitHub | EmonLibCM by Trystan Lea, Glyn Hudson, Brian Orpin and Ivan Kravets.

Please also see this project's companion Medium article [Energy Management Using Real-Time Non-Intrusive Load Monitoring](https://towardsdatascience.com/energy-management-using-real-time-non-intrusive-load-monitoring-3c9b0b4c8291).

## Appendix

### Photograph of Prototype

A photograph of an early version of my prototype system is shown below.

![Alt text](./img/early-prototype.png?raw=true "Early Prototype")

### Analog Signal Conditioning Schematic

The schematic for the Analog Signal Conditioning circuitry is shown below.

![Alt text](./img/pan-ard-inf-v1.1.jpg?raw=true "Analog Signal Conditioning Schematic")

### Model Training Results

Typical model performance evaluated against the metrics described above are as follows.

```text
### Dishwasher ###
2022–06–05 12:48:09,322 [INFO ] Appliance target is: dishwasher
2022–06–05 12:48:09,322 [INFO ] File for test: dishwasher_test_H20.csv
2022–06–05 12:48:09,322 [INFO ] Loading from: ./dataset_management/refit/dishwasher/dishwasher_test_H20.csv
2022–06–05 12:48:10,010 [INFO ] There are 5.169M test samples.
2022–06–05 12:48:10,015 [INFO ] Loading saved model from ./models/dishwasher/checkpoints.
2022–06–05 12:48:35,045 [INFO ] aggregate_mean: 522
2022–06–05 12:48:35,046 [INFO ] aggregate_std: 814
2022–06–05 12:48:35,046 [INFO ] appliance_mean: 700
2022–06–05 12:48:35,046 [INFO ] appliance_std: 1000
2022–06–05 12:48:35,121 [INFO ] true positives=5168007.0
2022–06–05 12:48:35,121 [INFO ] false negatives=0.0
2022–06–05 12:48:35,121 [INFO ] recall=1.0
2022–06–05 12:48:35,173 [INFO ] true positives=5168007.0
2022–06–05 12:48:35,173 [INFO ] false positives=0.0
2022–06–05 12:48:35,173 [INFO ] precision=1.0
2022–06–05 12:48:35,173 [INFO ] F1:1.0
2022–06–05 12:48:35,184 [INFO ] NDE:0.45971032977104187
2022–06–05 12:48:35,657 [INFO ] 
MAE: 10.477645874023438
 -std: 104.59112548828125
 -min: 0.0
 -max: 3588.0
 -q1: 0.0
 -median: 0.0
 -q2: 0.78582763671875
2022–06–05 12:48:35,675 [INFO ] SAE: 0.1882956475019455
2022–06–05 12:48:35,691 [INFO ] Energy per Day: 145.72999548734632

### Microwave ###
2022-06-05 18:14:05,471 [INFO ]  Appliance target is: microwave
2022-06-05 18:14:05,471 [INFO ]  File for test: microwave_test_H4.csv
2022-06-05 18:14:05,471 [INFO ]  Loading from: ./dataset_management/refit/microwave/microwave_test_H4.csv
2022-06-05 18:14:06,476 [INFO ]  There are 6.761M test samples.
2022-06-05 18:14:06,482 [INFO ]  Loading saved model from ./models/microwave/checkpoints.
2022-06-05 18:14:38,385 [INFO ]  aggregate_mean: 522
2022-06-05 18:14:38,385 [INFO ]  aggregate_std: 814
2022-06-05 18:14:38,385 [INFO ]  appliance_mean: 500
2022-06-05 18:14:38,385 [INFO ]  appliance_std: 800
2022-06-05 18:14:38,478 [INFO ]  true positives=6759913.0
2022-06-05 18:14:38,478 [INFO ]  false negatives=0.0
2022-06-05 18:14:38,478 [INFO ]  recall=1.0
2022-06-05 18:14:38,549 [INFO ]  true positives=6759913.0
2022-06-05 18:14:38,549 [INFO ]  false positives=0.0
2022-06-05 18:14:38,549 [INFO ]  precision=1.0
2022-06-05 18:14:38,549 [INFO ]  F1:1.0
2022-06-05 18:14:38,568 [INFO ]  NDE:0.6228251457214355
2022-06-05 18:14:39,469 [INFO ]  
MAE: 7.6666789054870605
    -std: 73.37799835205078
    -min: 0.0
    -max: 3591.474365234375
    -q1: 0.757080078125
    -median: 1.178070068359375
    -q2: 1.459686279296875
2022-06-05 18:14:39,493 [INFO ]  SAE: 0.2528369128704071
2022-06-05 18:14:39,512 [INFO ]  Energy per Day: 99.00535584592438

### Fridge ###
2022-06-06 05:14:39,830 [INFO ]  Appliance target is: fridge
2022-06-06 05:14:39,830 [INFO ]  File for test: fridge_test_H15.csv
2022-06-06 05:14:39,830 [INFO ]  Loading from: ./dataset_management/refit/fridge/fridge_test_H15.csv
2022-06-06 05:14:40,671 [INFO ]  There are 6.226M test samples.
2022-06-06 05:14:40,677 [INFO ]  Loading saved model from ./models/fridge/checkpoints.
2022-06-06 05:15:11,539 [INFO ]  aggregate_mean: 522
2022-06-06 05:15:11,539 [INFO ]  aggregate_std: 814
2022-06-06 05:15:11,539 [INFO ]  appliance_mean: 200
2022-06-06 05:15:11,539 [INFO ]  appliance_std: 400
2022-06-06 05:15:11,649 [INFO ]  true positives=6225098.0
2022-06-06 05:15:11,649 [INFO ]  false negatives=0.0
2022-06-06 05:15:11,649 [INFO ]  recall=1.0
2022-06-06 05:15:11,713 [INFO ]  true positives=6225098.0
2022-06-06 05:15:11,713 [INFO ]  false positives=0.0
2022-06-06 05:15:11,713 [INFO ]  precision=1.0
2022-06-06 05:15:11,713 [INFO ]  F1:1.0
2022-06-06 05:15:11,728 [INFO ]  NDE:0.390367716550827
2022-06-06 05:15:12,732 [INFO ]  
MAE: 18.173030853271484
    -std: 22.19791030883789
    -min: 0.0
    -max: 2045.119873046875
    -q1: 5.2667236328125
    -median: 12.299388885498047
    -q2: 24.688186645507812
2022-06-06 05:15:12,754 [INFO ]  SAE: 0.3662513792514801
2022-06-06 05:15:12,774 [INFO ]  Energy per Day: 219.63657335193759

### Washing Machine ###
2022-06-05 05:49:17,614 [INFO ]  Appliance target is: washingmachine
2022-06-05 05:49:17,614 [INFO ]  File for test: washingmachine_test_H8.csv
2022-06-05 05:49:17,614 [INFO ]  Loading from: ./dataset_management/refit/washingmachine/washingmachine_test_H8.csv
2022-06-05 05:49:18,762 [INFO ]  There are 6.118M test samples.
2022-06-05 05:49:18,767 [INFO ]  Loading saved model from ./models/washingmachine/checkpoints.
2022-06-05 05:49:47,965 [INFO ]  aggregate_mean: 522
2022-06-05 05:49:47,965 [INFO ]  aggregate_std: 814
2022-06-05 05:49:47,965 [INFO ]  appliance_mean: 400
2022-06-05 05:49:47,965 [INFO ]  appliance_std: 700
2022-06-05 05:49:48,054 [INFO ]  true positives=6117871.0
2022-06-05 05:49:48,055 [INFO ]  false negatives=0.0
2022-06-05 05:49:48,055 [INFO ]  recall=1.0
2022-06-05 05:49:48,115 [INFO ]  true positives=6117871.0
2022-06-05 05:49:48,115 [INFO ]  false positives=0.0
2022-06-05 05:49:48,115 [INFO ]  precision=1.0
2022-06-05 05:49:48,115 [INFO ]  F1:1.0
2022-06-05 05:49:48,128 [INFO ]  NDE:0.4052383601665497
2022-06-05 05:49:48,835 [INFO ]  
MAE: 20.846961975097656
    -std: 155.17930603027344
    -min: 0.0
    -max: 3972.0
    -q1: 3.0517578125e-05
    -median: 0.25152587890625
    -q2: 1.6888427734375
2022-06-05 05:49:48,856 [INFO ]  SAE: 0.3226347267627716
2022-06-05 05:49:48,876 [INFO ]  Energy per Day: 346.94591079354274

### Kettle ###
2022-05-25 15:19:15,366 [INFO ]  Appliance target is: kettle
2022-05-25 15:19:15,366 [INFO ]  File for test: kettle_test_H2.csv
2022-05-25 15:19:15,366 [INFO ]  Loading from: ./dataset_management/refit/kettle/kettle_test_H2.csv
2022-05-25 15:19:16,109 [INFO ]  There are 5.734M test samples.
2022-05-25 15:19:16,115 [INFO ]  Loading saved model from ./models/kettle/checkpoints.
2022-05-25 15:19:44,459 [INFO ]  aggregate_mean: 522
2022-05-25 15:19:44,459 [INFO ]  aggregate_std: 814
2022-05-25 15:19:44,459 [INFO ]  appliance_mean: 700
2022-05-25 15:19:44,459 [INFO ]  appliance_std: 1000
2022-05-25 15:19:44,540 [INFO ]  true positives=5732928.0
2022-05-25 15:19:44,540 [INFO ]  false negatives=0.0
2022-05-25 15:19:44,540 [INFO ]  recall=1.0
2022-05-25 15:19:44,597 [INFO ]  true positives=5732928.0
2022-05-25 15:19:44,597 [INFO ]  false positives=0.0
2022-05-25 15:19:44,597 [INFO ]  precision=1.0
2022-05-25 15:19:44,597 [INFO ]  F1:1.0
2022-05-25 15:19:44,610 [INFO ]  NDE:0.2578023374080658
2022-05-25 15:19:45,204 [INFO ]  
MAE: 18.681659698486328
    -std: 125.82673645019531
    -min: 0.0
    -max: 3734.59228515625
    -q1: 0.0
    -median: 0.20867919921875
    -q2: 1.63885498046875
2022-05-25 15:19:45,224 [INFO ]  SAE: 0.2403179109096527
2022-05-25 15:19:45,243 [INFO ]  Energy per Day: 155.25137268110353
```
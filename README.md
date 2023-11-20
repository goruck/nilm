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

Energy disaggregation is a highly underdetermined and a single-channel Blind Source Separation (BSS) problem² which makes it difficult to obtain accurate predictions. Let $M$ be the number of household appliances and $i$ be the index referring to the $i$-th appliance. The aggregate power consumption $x$ at a given time $i$ is the sum of the power consumption of all appliances $M$, denoted by $y_i\forall{i=1,...,M}$. Therefore, the total power consumption $x$ at a given time $t$ can expressed by the equation below. 

$$x(t)=\sum_{i=1}^My_i(t)+\epsilon_{noise}(t) \tag{1}$$

Where $\epsilon_{noise}$ is a noise term. The goal of this project is to solve the inverse problem and estimate the appliance power consumption $y_i$, given the aggregate power signal $x$, and to do so in a manner suitable for deployment at the edge. 

Past approaches have included factorial hidden Markov models (FHMM)¹ and various event-based methods with some success⁶. You can also solve the single-channel BSS problem can by using sequence-to-sequence (seq2seq) learning with neural networks, and it can applied to the NILM problem using transformers, convolutional and recurrent neural networks³. Seq2seq learning involves training a neural network to map between an input time series, such as the aggregate power readings the case of NILM, and a output sequence, such as the estimated energy consumption of a single appliance. A sliding input window is typically used to training the network which generates a corresponding window of the output. This method produces multiple predictions for each appliance in the output so an average of the predictions is used for the final result. Some of the predictions will be more accurate than others, especially those near the midpoint of the input sliding window. The averaging will tend to lower the overall accuracy of the predictions.

Some of the disadvantages of of seq2seq leaning can mitigated by sequence-to-point learning (seq2point) for single-channel BSS⁴. You also use a sliding input signal window in this approach, however the network is trained to predict the output signal only at the midpoint of the window which makes the prediction problem easier on the network, leading to more accurate results.

I selected the seq2point learning approach for my prototype system and my implementation was inspired and guided by work described by Michele D'Incecco, Stefano Squartini and Mingjun Zhong⁵. I developed a variety of seq2point learning models using Tensorflow as shown in the Python module [define_models.py](./ml/define_models.py) but focussed my work on the models `transformer` and `cnn`.

### Neural Network Models

The ```cnn``` model is depicted below for an input sequence length of 599 samples. The model generally follows traditional cnn concepts from vision use cases where several convolutional layers are used to extract features from the input power sequence at gradually finer details as the input traverses through the network. These features are the on-off patterns of the appliances as well as their power consumption levels. Max pooling is used to manage the complexity of the model after each convolutional layer. Finally, dense layers are used to output the final single point power consumption estimate for the window which is de-normalized before using in downstream processing. There are about 40 million parameters in this model using the default values.

![Alt text](./img/cnn_model_plot.png?raw=true "cnn model plot")

The ```transformer``` model is depicted below for an input sequence length of 599 samples where the transformer block is a Bert-style encoder. The input sequence is first passed through a convolutional layer to expand it into a latent space which is analogous to the feature extraction in the cnn model case. These features are pooled and L2 normalized to reduce model complexity and to mitigate the effects of outliers. Next, the sequence features are processed by a Bert-style transformer lineup which includes positional embedding and transformer blocks that applies importance weighting. The output of the encoder is decoded by several layers which are comprised of relative position embedding which applies symmetric weights around the mid-point of the signal, average pooling which reduces the sequence to a single value per feature and then finally dense layers that output the final single point estimated power value for the window which again is de-normalized for downstream processing. There are about six million parameters in this model using the default values.

![Alt text](./img/transformer_model_plot.png?raw=true "transformer model plot")

The Bert-style transformer encoder is depicted below.

![Alt text](./img/transformer_block_plot.png?raw=true "bert-style encoder")

### Datasets

There are a number of large-scale publicly available datasets specifically designed to address the NILM problem which were captured in household buildings from various countries. The table⁷ below shows several of the most widely used.

![Alt text](./img/nilm-datasets.png?raw=true "NILM Datasets (Oliver Parson, et al.⁷)")

The datasets generally include many 10’s of millions of active power, reactive power, current, and voltage samples but with different sampling frequencies which requires you to pre-process the data before use. Most NILM algorithms utilize only real (aka active or true) power data. Five appliances are usually considered for energy disaggregation research which are kettle, microwave, fridge, dish washer and washing machine. These are the appliances I considered for my prototype and following the work of Michele DIncecco, et al.⁵, I mainly focused on the REFIT⁸ dataset but will eventually include UK-DALE and REDD.

Note that these datasets are typically very imbalanced because the majority of the time an appliance is in the off state. 

### Model Training

I used TensorFlow to train and test the model. All code associated with this section can be found on the Machine Learning section of the project’s GitHub, NILM⁹. The seq2point learning models for the appliances were trained individually on z-score standardized REFIT data or normalized to $[0, P_m]$, where $P_m$ is the maximum power consumption of an appliance in its active state. Using normalized data tended to give the best model performance so it is used by default.

I used the following metrics to evaluate the model’s performance. You can view the code that calculates these metrics [here](./ml/nilm_metric.py).

* Mean absolute error ($MAE$), which evaluates the absolute difference between the prediction and the ground truth power at every time point and calculates the mean value, as defined by the equation below.

$$MAE = \frac{1}{N}\sum_{i=1}^{N}|\hat{x_i}-x_i|\tag{2}$$

* Normalized signal aggregate error ($SAE$), which indicates the relative error of the total energy. Denote $r$ as the total energy consumption of the appliance and $\hat{r}$ as the predicted total energy, then SAE is defined per the equation below.

$$SAE = \frac{|\hat{r} - r|}{r}\tag{3}$$

* Energy per Day ($EpD$) which measures the predicted energy used in a day, useful when the household users are interested in the total energy consumed in a period. Denote $D$ as the total number of days and $e=\sum_{t}e_t$ as the appliance energy consumed in a day, then EpD is defined per the equation below.

$$EpD = \frac{1}{D}\sum_{n=1}^{D}e\tag{4}$$

* Normalized disaggregation error ($NDE$) which measures the normalized error of the squared difference between the prediction and the ground truth power of the appliances, as defined by the equation below.

$$NDE = \frac{\sum_{i,t}(x_{i,t}-\hat{x_{i,t}})^2}{\sum_{i,t}x_{i,t}^2}\tag{5}$$

I also used accuracy ($ACC$), F1-score ($F1$) and Matthew’s correlation coefficient ($MCC$) to assess if the model can perform well with the severely imbalanced datasets used to train and test the model. These metrics depend on the on-off status of the device and are computed using the parameters in the [common.py](./ml/common.py) module. $ACC$ is equal to the number of correctly predicted time points over the test dataset. $F1$ and $MCC$ are computed according to the equations below where $TP$ stands for true positives, $TN$ stands for true negatives, $FP$ stands for false positives and $FN$ stands for false negatives.

$$F1=\frac{TP}{TP+\frac{1}{2}(FP+FN)}\tag{6}$$

$$MCC=\frac{TN \times TP+FN \times FP }{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}\tag{7}$$

$MAE$, $SAE$, $NDE$ and $EpD_e$ (defined as $ 100\% \times (EpD_{predicted} - EpD_{ground\_truth}) / EpD_{ground\_truth}$) reflect the model's ability to correctly predict the appliance energy consumption levels. $F1$ and $MCC$ indicates the model's ability to correctly predict appliance activations using imbalanced classes. $ACC$ is less useful in this application because most of the time the model will correctly predict the appliance is off which dominates the dataset.

A sliding window of 599 samples of the aggregate real power consumption signal is used as inputs to seq2point model and the midpoints of the corresponding windows of the appliances are used as targets. You can see how the samples and targets are generated in the `get_window_generator` function in the [common.py](./ml/common.py) module.

You can see the code I used to train the model in [train_distributed.py](./ml/train_distributed.py) which uses the `tf.distribute.MirroredStrategy()` distributed training strategy. I used the Keras Adam optimizer and to reduce over-fitting, early stopping is used. The training code can be configured to train the seq2point model from scratch, or given a fitted model, prune it or fine tune it with quantization aware training (QAT), both of which can improve inference performance especially on edge hardware.

The key hyper-parameters for training and the optimizer are summarized below.
* Input Window Size: 599 samples
* Global Batch size: 1024 samples.
* From scratch Learning Rate: 1e-04
* QAT and Prune Learning Rate: 1e-05
* Adam Optimizer: beta_1=0.9, beta_2=0.999, epsilon=1e-08.
* Early Stopping Criteria: 6 epochs.

The loss function<sup>11</sup> shown in the equation below is used to compute training gradients and to evaluate validation loss on a per-batch basis. It consists of a combination of Mean Squared Error, Binary Cross-Entropy and Mean Absolute Error losses, averaged over distributed model replica batches.

$$L(x, s) = (\hat{x} - x)^2 -(\hat{s}\log{s}+(1-\hat{s})\log{(1-s)}) + (\lambda|\hat{x}-x|, \hat{x}\in\set{O})\tag{8}$$

Where $x, \hat{x}\in[0, 1]$ are the ground truth and predicted power usage single point values divided by the maximum power limit per appliance and $s, \hat{s}\in\set{0, 1}$ are the appliance state label and prediction, and $O$ is the set of predictions when either the status label is on or the prediction is incorrect. The hyperparameter $\lambda$ tunes the absolute loss term on an a per-appliance basis. 

You can find the training results for each appliance in the [models](./ml/models/) folder. Typical performance metrics for the `cnn` model are shown in the table below.

|Appliance|$F1\uparrow$|$MCC\uparrow$|$ACC\uparrow$|$MAE$ $(W)$ $\downarrow$|$SAE\downarrow$|$NDE\downarrow$|$EpD_e\thinspace(\%)\downarrow$|
| --- | --- | --- | --- | --- | --- | --- | --- |
|kettle|0.7199|0.7298|0.9960|8.723|0.3091|0.4705|-30.91|
|microwave|0.6328|0.6303|0.9950|6.070|0.2102|0.6876|-21.02|
|fridge|0.7978|0.7002|0.8620|14.71|0.1471|0.4137|14.71|
|dishwasher|0.5825|0.6281|0.9790|5.007|0.0193|0.3450|1.926|
|washingmachine|0.8478|0.8441|0.9893|14.75|0.2929|0.3470|-29.29|

Typical performance metrics for the `transformer` model are shown in the table below.

|Appliance|$F1\uparrow$|$MCC\uparrow$|$ACC\uparrow$|$MAE$ $(W)$ $\downarrow$|$SAE\downarrow$|$NDE\downarrow$|$EpD_e\thinspace(\%)\downarrow$|
| --- | --- | --- | --- | --- | --- | --- | --- |
|kettle|0.8177|0.8176|0.9967|7.264|0.1348|0.3760|13.48|
|microwave|0.6530|0.6506|0.9952|6.428|0.1454|0.7496|-14.54|
|fridge|0.8138|0.7262|0.8799|11.02|0.0559|0.3743|5.590|
|dishwasher|0.6914|0.7119|0.9873|5.373|0.0690|0.3906|-6.904|
|washingmachine|0.8435|0.8420|0.9886|14.19|0.2440|0.2440|-24.40|

Average metrics across all appliances for both model architectures are compared in the table below.

|Architecture|$\overline{F1}\uparrow$|$\overline{MCC}\uparrow$|$\overline{ACC}\uparrow$|$\overline{MAE}$ $(W)$ $\downarrow$|$\overline{SAE}\downarrow$|$\overline{NDE}\downarrow$|$\overline{\|EpD_e\|}\thinspace(\%)\downarrow$|
| --- | --- | --- | --- | --- | --- | --- | --- |
|```cnn```|0.7161|0.7065|0.9643|9.852|0.1957|0.4528|19.57|
|```transformer```|0.7639|0.7497|0.9695|8.855|0.1298|0.3781|12.98|

You can see that the ```cnn``` and ```transformer``` models have similar performance even though the latter has about six times fewer parameters than the former. However, each ```transformer``` training step takes about seven times longer than ```cnn``` due to the `transformer` model's use of self-attention which has $O(n^2)$ complexity as compared to the `cnn` model's $O(n)$, where $n$ is the input sequence length. On the basis on training (and inference) efficiency, you can see that ```cnn``` is preferable with little loss in model performance.

You can find a variety of performance metrics in the literature to compare these results against, here are two examples. The middle column in the table<sup>5</sup> below shows another cnn-based model's performance. You can see the results of this project compare very favorably. 

![Alt text](./img/cnn_results_baseline.png?raw=true "Transfer Learning for Non-Intrusive Load Monitoring by Michele D'Incecco, Stefano Squartini and Mingjun Zhong[5]")

The table<sup>12</sup> below shows the results from two transformer-based models (BERT4NILM and ELECTRIcity) and three other architectures. Although the data is incomplete, again this project's transformer-based model results compare favorably. 

![Alt text](img/transformer_results_baseline.png?raw=true "ELECTRIcity: An Efficient Transformer for Non-Intrusive Load Monitoring by Stavros Sykiotis, Maria Kaselimi ,Anastasios Doulamis and Nikolaos Doulamis[12].")

### Model Quantization

I quantized the ```cnn``` model’s weights and activation functions from Float32 to INT8 using TensorFlow Lite to improve inference performance on edge hardware, including the Raspberry Pi and the Google Edge TPU. Only the weights for the ```transformer``` model were quantized to INT8, the activations needed to be kept in Float32 to maintain accptable accuracy. See [convert_keras_to_tflite.py](./ml/convert_keras_to_tflite.py) for the code that does this quantization which also uses [TFLite's quantization debugger](https://www.tensorflow.org/lite/performance/quantization_debugger) to check how well each layer in the model was quantized. The quantized inference results are shown in the tables below, where $R_{x86}$ is the inference rate on a 3.8 GHz x86 machine using eight tflite interpreter threads with XNNPACK and $R_{\pi}$ is the inference rate on a Raspberry Pi 4. You will observed a slight degradation in performance after quantization but this is acceptable for most use cases.

The quantized results for the ```cnn``` model are shown in the table below.

|Appliance|$F1\uparrow$|$MCC\uparrow$|$ACC\uparrow$|$MAE$ $(W)$ $\downarrow$|$SAE\downarrow$|$NDE\downarrow$|$EpD_e$ ($\%$)|$R_{x86}$ ($Hz$)|$R_{\pi}$ ($Hz$)|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|kettle|0.6313|0.6510|0.9957|9.220|0.4034|0.5768|-40.34|5081|x|
|microwave|0.6221|0.6283|0.9943|7.652|0.4811|0.6841|-48.11|5128|x|
|fridge|0.5980|0.4354|0.7665|19.17|0.1101|0.7746|-10.88|5108|x|
|dishwasher|0.6645|0.6775|0.9842|5.874|0.0167|0.3895|-1.668|5050|x|
|washingmachine|0.8910|0.8872|0.9926|15.08|0.3796|0.3442|-37.96|5094|x|

The quantized results for the ```transformer``` model are shown in the table below (TBD).

|Appliance|$F1\uparrow$|$MCC\uparrow$|$ACC\uparrow$|$MAE$ $(W)$ $\downarrow$|$SAE\downarrow$|$NDE\downarrow$|$EpD_e$ ($\%$)|$R_{x86}$ ($Hz$)|$R_{\pi}$ ($Hz$)|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|kettle|0.6313|0.6510|0.9957|9.220|0.4034|0.5768|-40.34|5081|x|
|microwave|0.6221|0.6283|0.9943|7.652|0.4811|0.6841|-48.11|5128|x|
|fridge|0.5980|0.4354|0.7665|19.17|0.1101|0.7746|-10.88|5108|x|
|dishwasher|0.6645|0.6775|0.9842|5.874|0.0167|0.3895|-1.668|5050|x|
|washingmachine|0.8910|0.8872|0.9926|15.08|0.3796|0.3442|-37.96|5094|x|

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
11. BERT4NILM: A Bidirectional Transformer Model for Non-Intrusive Load Monitoring by Zhenrui Yue, et. al.
12. ELECTRIcity: An Efficient Transformer for Non-Intrusive Load Monitoring by Stavros Sykiotis, Maria Kaselimi ,Anastasios Doulamis and Nikolaos Doulamis.

Please also see this project's companion Medium article [Energy Management Using Real-Time Non-Intrusive Load Monitoring](https://towardsdatascience.com/energy-management-using-real-time-non-intrusive-load-monitoring-3c9b0b4c8291).

## Appendix

### Photograph of Prototype

A photograph of an early version of my prototype system is shown below.

![Alt text](./img/early-prototype.png?raw=true "Early Prototype")

### Analog Signal Conditioning Schematic

The schematic for the Analog Signal Conditioning circuitry is shown below.

![Alt text](./img/pan-ard-inf-v1.2.png?raw=true "Analog Signal Conditioning Schematic")
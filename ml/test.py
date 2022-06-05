"""
Test a neural network to perform energy disaggregation,
i.e., given a sequence of electricity mains reading,
the algorithm separates the mains into appliances.

References:
(1) Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton.
``Sequence-to-point learning with neural networks for nonintrusive load monitoring."
Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), Feb. 2-7, 2018.

(2) https://arxiv.org/abs/1902.08835

(3) https://github.com/MingjunZhong/transferNILM.

Copyright (c) 2022 Lindo St. Angel
"""

import os
import argparse
import socket

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from logger import log
import nilm_metric as nm
from common import WindowGenerator, load_dataset, params_appliance

def normalize(test_set_x, test_set_y):
    """Normalize or standardize datasets.
    
    Currently not in use.
    """
    # Normalize
    import numpy as np
    # Compute aggregate statistics.
    agg_mean = 522
    agg_std = 814
    print(f'agg mean: {agg_mean}, agg std: {agg_std}')
    agg_median = 308.0
    agg_quartile1 = 173.0
    agg_quartile3 = 584.0
    log(f'agg median: {agg_median},agg q1: {agg_quartile1}, agg q3: {agg_quartile3}')
    # Compute appliance statistics.
    app_mean = 200
    app_std = 400
    log(f'app mean: {app_mean}, app std: {app_std}')
    app_median = 5.0
    app_quartile1 = 1.0
    app_quartile3 = 87.0
    log(f'app median: {app_median}, app q1: {app_quartile1}, app q3: {app_quartile3}')
    def z_norm(dataset, mean, std):
        return (dataset - mean) / std
    def robust_scaler(dataset, median, quartile1, quartile3):
        return (dataset - median) / (quartile3 - quartile1)
    return z_norm(test_set_x, agg_mean, agg_std), z_norm(test_set_y, app_mean, app_std)

def get_arguments():
    parser = argparse.ArgumentParser(description='Predict appliance\
                                     given a trained neural network\
                                     for energy disaggregation -\
                                     network input = mains window;\
                                     network target = the states of\
                                     the target appliance.')
    parser.add_argument('--appliance_name',
                        type=str,
                        default='kettle',
                        help='the name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default='./dataset_management/refit',
                        help='this is the directory to the test data')
    parser.add_argument('--trained_model_dir',
                        type=str,
                        default='./models',
                        help='this is the directory to the trained models')
    parser.add_argument('--ckpt_dir',
                        type=str,
                        default='checkpoints',
                        help='directory name of model checkpoint')
    parser.add_argument('--save_results_dir',
                        type=str,
                        default='./results',
                        help='this is the directory to save the predictions')
    parser.add_argument('--test_type',
                        type=str,
                        default='test',
                        help='Type of the test set to load: \
                            test -- test on the proper test set;\
                            train -- test on a already prepared slice of the train set;\
                            val -- test on the validation set;\
                            uk -- test on UK-DALE;\
                            redd -- test on REDD.')
    parser.add_argument("--transfer", action='store_true',
                        help="If set, use a pre-trained CNN (True) or not (False).")
    parser.add_argument('--plot', action='store_true',
                        help='If set, plot the predicted appliance against ground truth.')
    parser.add_argument('--cnn',
                        type=str,
                        default='kettle',
                        help='The trained CNN for the appliance to load.')
    parser.add_argument('--crop',
                        type=int,
                        default=None,
                        help='To use part of the dataset for testing.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1000,
                        help='Sets mini-batch size.')
    parser.set_defaults(plot=False)
    parser.set_defaults(transfer=False)
    return parser.parse_args()

def find_test_filename(datadir, appliance_name) -> str:
    for filename in os.listdir(os.path.join(datadir, appliance_name)):
        if args.test_type == 'train' and 'TRAIN' in filename.upper():
            test_filename = filename
            break
        elif args.test_type == 'uk' and 'UK' in filename.upper():
            test_filename = filename
            break
        elif args.test_type == 'redd' and 'REDD' in filename.upper():
            test_filename = filename
            break
        elif args.test_type == 'test' and 'TEST' in\
                filename.upper() and 'TRAIN' not in filename.upper() and 'UK' not in filename.upper():
            test_filename = filename
            break
        elif args.test_type == 'val' and 'VALIDATION' in filename.upper():
            test_filename = filename
            break
    return test_filename

if __name__ == '__main__':
    log(f'Machine name: {socket.gethostname()}')
    args = get_arguments()
    log('Arguments: ')
    log(args)

    appliance_name = args.appliance_name
    log('Appliance target is: ' + appliance_name)

    test_filename = find_test_filename(args.datadir, appliance_name)
    log('File for test: ' + test_filename)
    test_file_path = os.path.join(args.datadir, appliance_name, test_filename)
    log('Loading from: ' + test_file_path)

    # offset parameter from window length
    offset = int(0.5 * (params_appliance[appliance_name]['windowlength'] - 1.0))

    test_set_x, test_set_y = load_dataset(test_file_path, args.crop)
    ts_size = test_set_x.size
    log(f'There are {ts_size/10**6:.3f}M test samples.')
    #if args.crop > ts_size:
        #log('(crop larger than dataset size, ignoring it)')

    # Ground truth is center of test target (y) windows.
    ground_truth = test_set_y[offset:-offset]

    test_provider = WindowGenerator(
        dataset=(test_set_x, None),
        train=False,
        shuffle=False)

    # Load best checkpoint from saved trained model for appliance.
    model_file_path = os.path.join(
        args.trained_model_dir, appliance_name, args.ckpt_dir)
    log(f'Loading saved model from {model_file_path}.')
    model = tf.keras.models.load_model(model_file_path)

    model.summary()

    test_prediction = model.predict(
        x=test_provider,
        verbose=1,
        workers=24,
        use_multiprocessing=True)

    # Parameters.
    max_power = params_appliance[appliance_name]['max_on_power']
    threshold = params_appliance[appliance_name]['on_power_threshold']
    aggregate_mean = 522
    aggregate_std = 814

    appliance_mean = params_appliance[appliance_name]['mean']
    appliance_std = params_appliance[appliance_name]['std']

    log('aggregate_mean: ' + str(aggregate_mean))
    log('aggregate_std: ' + str(aggregate_std))
    log('appliance_mean: ' + str(appliance_mean))
    log('appliance_std: ' + str(appliance_std))

    prediction = test_prediction * appliance_std + appliance_mean
    prediction[prediction <= 0.0] = 0.0
    ground_truth = ground_truth * appliance_std + appliance_mean

    # Metric evaluation.
    sample_second = 8.0  # sample period is 8 seconds
    log('F1:{0}'.format(nm.get_F1(ground_truth.flatten(), prediction.flatten(), threshold)))
    log('NDE:{0}'.format(nm.get_nde(ground_truth.flatten(), prediction.flatten())))
    log('\nMAE: {:}\n    -std: {:}\n    -min: {:}\n    -max: {:}\n    -q1: {:}\n    -median: {:}\n    -q2: {:}'
        .format(*nm.get_abs_error(ground_truth.flatten(), prediction.flatten())))
    log('SAE: {:}'.format(nm.get_sae(ground_truth.flatten(), prediction.flatten(), sample_second)))
    log('Energy per Day: {:}'.format(nm.get_Epd(ground_truth.flatten(), prediction.flatten(), sample_second)))

    # Save results.
    savemains = test_set_x.flatten() * aggregate_std + aggregate_mean
    savegt = ground_truth
    savepred = prediction.flatten()

    if args.transfer:
        save_name = args.save_results_dir + '/' + appliance_name + '/' + test_filename + '_transf_' + args.cnn
    else:
        save_name = args.save_results_dir + '/' + appliance_name + '/' + test_filename
    if not os.path.exists(save_name):
        os.makedirs(save_name)

    np.save(save_name + '_pred.npy', savepred)
    np.save(save_name + '_gt.npy', savegt)
    np.save(save_name + '_mains.npy', savemains)

    log('size: x={0}, y={0}, gt={0}'
        .format(np.shape(savemains), np.shape(savepred), np.shape(savegt)))

    # Plot.
    if args.plot:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(savemains[offset:-offset], color='#7f7f7f', linewidth=1.8)
        ax1.plot(ground_truth, color='#d62728', linewidth=1.6)
        ax1.plot(prediction,
                 color='#1f77b4',
                 #marker='o',
                 linewidth=1.5)
        ax1.grid()
        ax1.set_title('Test results on {:}'
            .format(test_filename), fontsize=16, fontweight='bold', y=1.08)
        ax1.set_ylabel('W')
        ax1.legend(['aggregate', 'ground truth', 'prediction'])
        #mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        plt.show()
        plt.close()
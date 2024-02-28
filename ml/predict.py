"""
Predict and analyze appliance type and power using novel data from my home.

Actual mains power is captured in a csv file by real-time telemetry code
running on a Raspberry Pi that also performs load disaggregation inference
using TFLite models converted from the trained TensorFlow floating point
models. These TensorFlow models are used here to also perform load disaggregation
and the results can be compared against the TFLite inferences. Ground truth power
consumption for the appliances are captured by another telemetry program
which can be used to evaluate both the TensorFlow and TFLite inference results.

Copyright (c) 2022~2024 Lindo St. Angel
"""

import os
import argparse
import socket
import glob
import sys
from collections import OrderedDict
from collections import Counter

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import pandas as pd

import common
from logger import Logger
from window_generator import WindowGenerator
from nilm_metric import NILMTestMetrics

# Input sample window length for all appliances
WINDOW_LENGTH = 599

# Names of all appliances that can be monitored.
DEFAULT_APPLIANCES = ['kettle', 'fridge', 'microwave', 'washingmachine', 'dishwasher']

# csv file name prefix of mains power and appliance predictions.
# These files are in the 'default_dataset_dir' identified below.
RT_DATA_FILE_PREFIX = 'samples'

# csv fine name of ground truth appliance power readings.
# These files are in the 'default_dataset_dir' identified below.
GT_DATA_FILE_PREFIX = 'appliance_energy_data_'

def get_arguments():
    """Get command line arguments."""
    default_trained_model_dir = '/home/lindo/Develop/nilm/ml/models'
    default_results_dir = '/home/lindo/Develop/nilm/ml/results'
    default_dataset_dir = '/home/lindo/Develop/nilm-datasets/my-house'
    default_panel_location = 'house'

    parser = argparse.ArgumentParser(
        description='Predict appliance type and power using novel data from my home.'
    )
    parser.add_argument(
        '--appliances',
        type=str,
        nargs='+',
        default=DEFAULT_APPLIANCES,
        help='name(s) of target appliance'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default=default_dataset_dir,
        help='novel dataset(s) directory'
    )
    parser.add_argument(
        '--panel',
        type=str,
        default=default_panel_location,
        help='sub-panel location'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=default_trained_model_dir,
        help='trained models directory'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=default_results_dir,
        help='directory to save the predictions'
    )
    parser.add_argument(
        '--crop',
        type=int,
        default=None,
        help='use part of the dataset for predictions'
    )
    parser.add_argument(
        '--model_arch',
        type=str,
        default='cnn',
        help='model architecture used for predictions'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='if set, show predicted appliance and mains power plots'
    )
    parser.add_argument(
        '--show_rt_preds',
        action='store_true',
        help='if set, show real-time predictions on plot'
    )
    parser.add_argument(
        '--show_gt',
        action='store_true',
        help='if set, show appliance ground truth plots'
    )
    parser.set_defaults(show_gt=False)
    parser.set_defaults(plot=False)
    parser.set_defaults(show_rt_preds=False)

    return parser.parse_args()

def predict(appliance_name:str, input_data:np.ndarray, model_dir:str, arch:str) -> np.ndarray:
    """Make appliance prediction and return de-normalized result."""
    # Preprocess input dataset.
    x = input_data.copy()
    train_agg_mean = common.params_appliance[appliance_name]['train_agg_mean']
    train_agg_std = common.params_appliance[appliance_name]['train_agg_std']
    agg_mean = common.ALT_AGGREGATE_MEAN if common.USE_ALT_STANDARDIZATION else train_agg_mean
    agg_std = common.ALT_AGGREGATE_STD if common.USE_ALT_STANDARDIZATION else train_agg_std
    x = (x - agg_mean) / agg_std

    test_provider = WindowGenerator(
        dataset=(x, None, None),
        train=False,
        shuffle=False
    )

    def gen():
        """Yields batches of data from test_provider for tf.data.Dataset."""
        for _, batch in enumerate(test_provider):
            yield batch
    test_dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, WINDOW_LENGTH, 1), dtype=tf.float32)
        )
    )

    model_filepath = os.path.join(model_dir, appliance_name, f'savemodel_{arch}')
    model = tf.keras.models.load_model(model_filepath)

    predictions = model.predict(
        test_dataset,
        steps=len(test_provider),
        verbose=1,
        workers=24,
        use_multiprocessing=True
    )

    return denormalize(predictions, appliance_name)

def denormalize(predictions:np.ndarray, appliance_name:str) -> np.ndarray:
    """De-normalize appliance power predictions."""
    if common.USE_APPLIANCE_NORMALIZATION:
        app_mean = 0
        app_std = common.params_appliance[appliance_name]['max_on_power']
    else:
        train_app_mean = common.params_appliance[appliance_name]['train_app_mean']
        train_app_std = common.params_appliance[appliance_name]['train_app_std']
        alt_app_mean = common.params_appliance[appliance_name]['alt_app_mean']
        alt_app_std = common.params_appliance[appliance_name]['alt_app_std']
        app_mean = alt_app_mean if common.USE_ALT_STANDARDIZATION else train_app_mean
        app_std = alt_app_std if common.USE_ALT_STANDARDIZATION else train_app_std

    # De-normalize.
    predictions = predictions.flatten() * app_std + app_mean
    # Remove negative energy predictions.
    predictions[predictions <= 0.0] = 0.0

    return predictions

def get_real_power(file_name, crop=None, zone='US/Pacific') -> pd.DataFrame:
    """Load real-time dataset and return total real power with datetimes."""
    dataframe = pd.read_csv(
        file_name,
        index_col=0,
        nrows=crop,
        usecols=['DT', 'W1', 'W2'],
        parse_dates=['DT'],
        date_format='ISO8601'
    )
    # Localize datetimes.
    dataframe = dataframe.tz_convert(tz=zone)
    # Convert NaN's into zero's
    dataframe = dataframe.fillna(0)
    # Compute total real power and insert into dataframe.
    # W1 and W2 are the real power in each phase.
    real_power = dataframe['W1'] + dataframe['W2']
    dataframe.insert(loc=0, column='real_power', value=real_power)
    # Remove W1 and W2 since they are no longer needed.
    dataframe = dataframe.drop(columns=['W1', 'W2'])
    # Adjustment for prediction timing.
    # Adjustment is done by moving samples later in time by a value equal
    # to the window center since the predict function places the prediction
    # at the beginning of a window of samples.
    window_center = int(0.5 * (WINDOW_LENGTH - 1.0))
    return dataframe.shift(periods=window_center, fill_value=0)

def get_ground_truth(file_name, crop=None, zone='US/Pacific') -> pd.DataFrame:
    """Load ground truth dataset and return appliance active power with datetimes."""
    dataframe = pd.read_csv(
        file_name,
        index_col=0,
        nrows=crop,
        usecols=['date', 'appliance', 'apower'],
        parse_dates=['date'],
        date_format='ISO8601'
    )
    # Localize datetimes.
    dataframe = dataframe.tz_convert(tz=zone)
    # Convert NaN's into zero's
    dataframe = dataframe.fillna(0)
    # Make columns of appliance data into rows.
    # Note apower is active power which is same as real power.
    dataframe = pd.pivot(dataframe, columns='appliance', values='apower')
    # Make column names consistent with convention.
    dataframe.rename(columns={'washing machine': 'washingmachine'}, inplace=True)
    # Downsample to match global sample rate.
    # Although the appliance ground truth power was sampled at 1 data point every 8 seconds,
    # each appliance was read consecutively by the logging program with about 1 second
    # between reads. Therefore the dataset has groups of appliance readings clustered
    # around 8 second intervals that this downsampling corrects for.
    resample_period = f'{common.SAMPLE_PERIOD}s'
    return dataframe.resample(resample_period).max()

def get_realtime_predictions(file_name, crop=None, zone='US/Pacific') -> pd.DataFrame:
    """Load appliance power predictions that were preformed in real-time with datetimes."""
    # Load real-time prediction dataset.
    dataframe = pd.read_csv(
        file_name,
        index_col=0, # make datetime the index
        nrows=crop,
        usecols=['DT']+DEFAULT_APPLIANCES,
        parse_dates=['DT'],
        date_format='ISO8601'
    )
    # Localize datetimes.
    dataframe = dataframe.tz_convert(tz=zone)
    # Convert NaN's into zero's
    dataframe = dataframe.fillna(0)
    # Adjustment for real-time prediction timing.
    # Adjustment is done by moving samples earlier in time by a value equal
    # to the window center since the real-time code places the prediction
    # at the end of a window of samples.
    window_center = int(0.5 * (WINDOW_LENGTH - 1.0))
    return dataframe.shift(periods=-window_center, fill_value=0)

def compute_metrics(
        ground_truth:np.ndarray,
        ground_truth_status:np.ndarray,
        pred:np.ndarray,
        pred_status:np.ndarray,
        log,
    ) -> None:
    """Assess performance of prediction vs ground truth."""
    metrics = NILMTestMetrics(
        target=ground_truth,
        target_status=ground_truth_status,
        prediction=pred,
        prediction_status=pred_status,
        sample_period=common.SAMPLE_PERIOD
    )
    log.log(f'True positives: {metrics.get_tp()}')
    log.log(f'True negatives: {metrics.get_tn()}')
    log.log(f'False positives: {metrics.get_fp()}')
    log.log(f'False negatives: {metrics.get_fn()}')
    log.log(f'Accuracy: {metrics.get_accuracy()}')
    log.log(f'MCC: {metrics.get_mcc()}')
    log.log(f'F1: {metrics.get_f1()}')
    log.log(f'MAE: {metrics.get_abs_error()["mean"]} (W)')
    log.log(f'NDE: {metrics.get_nde()}')
    log.log(f'SAE: {metrics.get_sae()}')
    epd_gt = metrics.get_epd(ground_truth * ground_truth_status, common.SAMPLE_PERIOD)
    log.log(f'Ground truth EPD: {epd_gt} (Wh)')
    epd_pred = metrics.get_epd(pred * pred_status, common.SAMPLE_PERIOD)
    log.log(f'Predicted EPD: {epd_pred} (Wh)')
    log.log(f'EPD Relative Error: {100.0 * (epd_pred - epd_gt) / epd_gt} (%)')

if __name__ == '__main__':
    args = get_arguments()

    logger = Logger(
        log_file_name=os.path.join(
            args.results_dir, args.panel, f'predict_{args.model_arch}.log'
        )
    )

    logger.log(f'Machine name: {socket.gethostname()}')
    logger.log('Arguments: ')
    logger.log(args)
    logger.log(
        'Using alt standardization.' if common.USE_ALT_STANDARDIZATION 
        else 'Using default standardization.'
    )

    # Get all mains real power samples.
    all_rt_files = glob.glob(
        os.path.join(args.datadir, args.panel) + f'/{RT_DATA_FILE_PREFIX}*.csv'
    )
    logger.log(f'Found mains real power files: {all_rt_files}.')
    rp_df_from_each_file = (get_real_power(f, crop=args.crop) for f in all_rt_files)
    rp_df = pd.concat(rp_df_from_each_file)
    rp_df.sort_index(inplace=True)
    logger.log(f'There are {rp_df.size/10**6:.3f}M real power samples.')
    # Get all ground truth datasets.
    all_gt_files = glob.glob(
        os.path.join(args.datadir, args.panel) + f'/{GT_DATA_FILE_PREFIX}*.csv'
    )
    logger.log(f'Found ground truth files: {all_gt_files}.')
    gt_df_from_each_file = (get_ground_truth(f, crop=args.crop) for f in all_gt_files)
    gt_df = pd.concat(gt_df_from_each_file)
    gt_df.sort_index(inplace=True)

    # Merge mains dataset with ground truth dataset.
    # Since mains was sampled asynchronously wrt ground truth, used asof merge.
    rp_df = pd.merge_asof(rp_df, gt_df, left_index=True, right_index=True)
    # Cover case where there is no ground truth for some appliances.
    all_appliance_counter = Counter(DEFAULT_APPLIANCES)
    gt_appliance_counter = Counter(gt_df.columns.to_list())
    if all_appliance_counter != gt_appliance_counter:
        missing_appliances = list(all_appliance_counter - gt_appliance_counter)
        logger.log(f'Adding missing ground truth appliances: {missing_appliances}.')
        rp_df.loc[:, missing_appliances] = np.nan

    # Compute and apply status to appliance powers in the merged mains and ground truth df.
    gt_appliance_powers = np.array(rp_df[args.appliances], dtype=np.float32)
    gt_status = np.array(
        [common.compute_status(gt_appliance_powers[:,i], a) for i, a in enumerate(args.appliances)]
    ).transpose()
    rp_df[args.appliances] *= gt_status

    # Get the mains (aka aggregate) real power and perform inference with it.
    aggregate = np.array(rp_df['real_power'], dtype=np.float32)
    logger.log('Making power predictions.')
    predicted_powers = {
        appliance: predict(
            appliance, aggregate, args.model_dir, args.model_arch
        ) for appliance in args.appliances
    }

    # Apply status to predictions.
    prediction_status = np.array(
        [common.compute_status(predicted_powers[a], a) for a in args.appliances]
    )
    for i, k in enumerate(predicted_powers):
        predicted_powers[k] *= prediction_status[i]

    # Get real-time prediction dataset and apply status.
    rt_preds_df_from_each_file = (get_realtime_predictions(f, crop=args.crop) for f in all_rt_files)
    rt_preds = pd.concat(rt_preds_df_from_each_file)
    rt_preds.sort_index(inplace=True)
    rt_preds = rt_preds[args.appliances] # cover case where subset of appliances are specified
    rt_appliance_powers = np.array(rt_preds, dtype=np.float32)
    rt_preds_status = np.array(
        [common.compute_status(rt_appliance_powers[:,i], a) for i, a in enumerate(args.appliances)]
    ).transpose()
    rt_preds *= rt_preds_status
    # Predicted dataset size is used to crop datasets since test_provider
    # defaults to outputting complete batches so there will be more mains
    # samples than predictions.
    # See `allow_partial_batches' parameter in window_generator.
    predicted_dataset_size = predicted_powers[args.appliances[0]].size

    # Compute metrics for predictions.
    logger.log('Evaluating performance metrics of predictions vs ground truth.')
    for i, a in enumerate(args.appliances):
        logger.log(f'\nMetrics for {a}:')
        compute_metrics(
            ground_truth=gt_appliance_powers[:,i][:predicted_dataset_size],
            ground_truth_status=gt_status[:,i][:predicted_dataset_size],
            pred=predicted_powers[a],
            pred_status=prediction_status[i],
            log=logger
        )
    logger.log('Evaluating performance metrics of real-time predictions vs ground truth.')
    for i, a in enumerate(args.appliances):
        logger.log(f'\nMetrics for {a}:')
        compute_metrics(
            ground_truth=gt_appliance_powers[:,i],
            ground_truth_status=gt_status[:,i],
            pred=rt_appliance_powers[:,i],
            pred_status=rt_preds_status[:,i],
            log=logger
        )

    ###
    ### Save and show appliance powers each in a single row of subplots.
    ###
    # Find max value in predictions for setting plot limits.
    max_pred = np.ceil(np.max(list(predicted_powers.values())))
    nrows = len(args.appliances) + 1
    fig, ax = plt.subplots(nrows=nrows, ncols=1, constrained_layout=False)
    ax[0].set_ylabel('Watts')
    ax[0].set_title('aggregate')
    ax[0].plot(rp_df['real_power'][:predicted_dataset_size], color='orange', linewidth=1.8)
    # Set friendly looking date and times on x-axis.
    locator = AutoDateLocator()
    ax[0].xaxis.set_major_locator(locator=locator)
    ax[0].xaxis.set_major_formatter(AutoDateFormatter(locator=locator))
    fig.autofmt_xdate()
    row = 1
    for appliance in args.appliances:
        ax[row].set_ylabel('Watts')
        ax[row].set_title(appliance)
        ax[row].set_ylim(0, max_pred)
        ax[row].plot(
            rp_df.index[:predicted_dataset_size],
            predicted_powers[appliance],
            color='red', linewidth=1.5, label='prediction'
        )
        if args.show_rt_preds:
            ax[row].plot(
                rt_preds[appliance], color='green',
                linewidth=1.5, label='real-time prediction'
            )
            ax[row].legend(loc='upper right')
        if args.show_gt:
            ax[row].plot(
                rp_df[appliance], color='blue',
                linewidth=1.5, label='ground_truth'
            )
            ax[row].legend(loc='upper right', fontsize='x-small')
        row+=1
    fig.suptitle(f'Prediction results for {args.panel}', fontsize=16, fontweight='bold')
    # Save and optionally show plot.
    plot_savepath = os.path.join(
        args.results_dir, args.panel,f'predict_{args.model_arch}_subplots.png'
    )
    plt.savefig(fname=plot_savepath)
    if args.plot:
        plt.show()
    plt.close()

    ###
    ### Show mains power and appliance power predictions on one plot.
    ###
    linestyles = OrderedDict(
        [('solid',               (0, ())),
        ('loosely dotted',      (0, (1, 10))),
        ('dotted',              (0, (1, 5))),
        ('densely dotted',      (0, (1, 1))),

        ('loosely dashed',      (0, (5, 10))),
        ('dashed',              (0, (5, 5))),
        ('densely dashed',      (0, (5, 1))),

        ('loosely dashdotted',  (0, (3, 10, 1, 10))),
        ('dashdotted',          (0, (3, 5, 1, 5))),
        ('densely dashdotted',  (0, (3, 1, 1, 1))),

        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
    )
    color_names = list(mcolors.TABLEAU_COLORS)
    line_styles = list(linestyles.values())
    fig, ax = plt.subplots()
    fig.suptitle(f'Prediction results for {args.panel}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Watts')
    ax.set_xlabel('Date-Time')
    # Plot mains power.
    ax.plot(
        rp_df['real_power'][:predicted_dataset_size],
        color=color_names[0], linewidth=1.8, linestyle=line_styles[0], label='Aggregate'
    )
    # Set friendly looking date and times on x-axis.
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator=locator)
    ax.xaxis.set_major_formatter(AutoDateFormatter(locator=locator))
    fig.autofmt_xdate()
    # Plot appliance powers.
    for i, appliance in enumerate(args.appliances):
        ax.plot(
            rp_df.index[:predicted_dataset_size],
            predicted_powers[appliance],
            color=color_names[i+1], linewidth=1.5, linestyle=line_styles[i+1], label=appliance
        )
    ax.legend(loc='upper right', fontsize='x-small')
    # Save and optionally show plot.
    plot_savepath = os.path.join(
        args.results_dir, args.panel, f'predict_{args.model_arch}_plot.png'
    )
    plt.savefig(fname=plot_savepath)
    if args.plot:
        plt.show()
    plt.close()

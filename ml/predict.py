"""
Predict appliance type and power using novel data from my home.

Copyright (c) 2022~2024 Lindo St. Angel
"""

import os
import argparse
import socket
from collections import OrderedDict

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

WINDOW_LENGTH = 599 # input sample window length for all appliances

DEFAULT_APPLIANCES = [
        'kettle', 'fridge', 'microwave', 'washingmachine', 'dishwasher'
    ]

def get_arguments():
    """Get command line arguments."""
    default_trained_model_dir = '/home/lindo/Develop/nilm/ml/models'
    default_results_dir = '/home/lindo/Develop/nilm/ml/results'
    default_dataset_dir = '/home/lindo/Develop/nilm-datasets/my-house'
    default_panel_location = 'house'
    default_rt_preds_filename = 'samples_02_20_23.csv'
    default_gt_filename= 'appliance_energy_data_022023.csv'

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
        '--rt_preds_filename',
        type=str,
        default=default_rt_preds_filename,
        help='real-time prediction dataset filename'
    )
    parser.add_argument(
        '--panel',
        type=str,
        default=default_panel_location,
        help='sub-panel location'
    )
    parser.add_argument(
        '--gt_filename',
        type=str, default=default_gt_filename,
        help='ground truth dataset filename'
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
        '--threshold_rt_preds',
        action='store_true',
        help='if set, apply power on thresholds to real-time predictions on plot'
    )
    parser.add_argument(
        '--show_gt',
        action='store_true',
        help='if set, show appliance ground truth plots'
    )
    parser.set_defaults(show_gt=False)
    parser.set_defaults(plot=False)
    parser.set_defaults(show_rt_preds=False)
    parser.set_defaults(threshold_rt_preds=False)

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
        nrows=crop,
        usecols=['DT', 'W1', 'W2'],
        na_filter=False,
        parse_dates=['DT'],
        date_format='ISO8601'
    )

    # Get datetimes and localize.
    df_dt = dataframe['DT'].dt.tz_convert(tz=zone)
    # Compute total real power.
    df_rp = dataframe['W1'] + dataframe['W2']

    return pd.concat([df_dt, df_rp], axis=1, keys=['datetime', 'real_power'])

def get_ground_truth(file_name, crop=None, zone='US/Pacific') -> pd.DataFrame:
    """Load ground truth dataset and return appliance active power with datetimes."""
    dataframe = pd.read_csv(
        file_name,
        nrows=crop,
        usecols=['date', 'appliance', 'apower'],
        na_filter=False,
        parse_dates=['date'],
        date_format='ISO8601'
    )

    # Get datetimes and localize.
    df_dt = dataframe['date'].dt.tz_convert(tz=zone)

    # Make columns of appliance data into rows.
    # Note apower is active power which is same as real power.
    df_ap = pd.pivot(dataframe, columns='appliance', values='apower')
    # Convert NaN's into zero's.
    df_ap = df_ap.fillna(0)

    # Concat datetimes and appliance active power into single dataframe.
    df_ap = pd.concat([df_dt, df_ap], axis=1)

    # Make column names consistent with convention.
    df_ap.rename(columns={'date': 'datetime', 'washing machine': 'washingmachine'}, inplace=True)

    # Downsample to match global sample rate.
    # Although the appliance ground truth power was sampled at 1 data point every 8 seconds,
    # each appliance was read consecutively by the logging program with about 1 second
    # between reads. Therefore the dataset has groups of appliance readings clustered
    # around 8 second intervals that this downsampling corrects for.
    resample_period = f'{common.SAMPLE_PERIOD}s'
    return df_ap.resample(resample_period, on='datetime').max()

def get_realtime_predictions(
        file_name, appliances, crop=None, apply_threshold=False
    ) -> pd.DataFrame:
    """Load appliance power predictions that were preformed in real-time."""
    # Load real-time prediction dataset.
    df_rt = pd.read_csv(file_name, usecols=DEFAULT_APPLIANCES, nrows=crop)
    df_rt = df_rt.fillna(0) # convert NaN's into zero's
    # Define real-time predictions columns to appliance names.
    # Select appliance prediction column then adjust for output timing.
    # Adjustment is simply moving samples earlier in time by
    # a WINDOW_LENGTH since the real-time code places the prediction
    # at the end of a window of samples.
    rt_predictions = {
        appliance: np.array(
            df_rt[[appliance]][WINDOW_LENGTH:], dtype=np.float32
        ) for appliance in appliances
    }

    if apply_threshold:
        for app in appliances:
            threshold = common.params_appliance[app]['on_power_threshold']
            rt_predictions[app][rt_predictions[app] <= threshold] = 0.0

    return rt_predictions

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

    # Calculate center of input window.
    # This is the location of the single point predicted appliance power.
    # This is used as an offset to locate the first and last points in the data plots.
    offset = int(0.5 * (WINDOW_LENGTH - 1.0))

    # Get mains real power samples and datetimes.
    rt_filename = os.path.join(args.datadir, args.panel, args.rt_preds_filename)
    rp_df = get_real_power(rt_filename, crop=args.crop)
    aggregate = np.array(rp_df['real_power'], dtype=np.float32)
    logger.log(f'There are {aggregate.size/10**6:.3f}M test samples.')

    # Get ground truth dataset.
    gt_filename = os.path.join(args.datadir, args.panel, args.gt_filename)
    gt = get_ground_truth(gt_filename)

    # Merge mains dataset with ground truth dataset.
    mains_gt_df = pd.merge_asof(rp_df, gt, on='datetime')

    logger.log('Making power predictions.')
    predicted_powers = {
        appliance: predict(
            appliance, aggregate, args.model_dir, args.model_arch
        ) for appliance in args.appliances
    }

    logger.log('Applying status to predictions.')
    for appliance in args.appliances:
        status = np.array(common.compute_status(predicted_powers[appliance], appliance))
        predicted_powers[appliance] *= status

    # Calculate metrics.
    logger.log('*** Metric Summary ***')
    aggregate_epd = NILMTestMetrics.get_epd(aggregate, common.SAMPLE_PERIOD)
    logger.log(f'Aggregate energy: {aggregate_epd/1000:.3f} kWh per day')
    for appliance in args.appliances:
        epd = NILMTestMetrics.get_epd(predicted_powers[appliance], common.SAMPLE_PERIOD)
        logger.log(f'{appliance} energy: {epd/1000:.3f} kWh per day')

    if args.show_rt_preds:
        rt_preds = get_realtime_predictions(
            rt_filename, args.appliances, args.crop, args.threshold
        )

    ###
    ### Save and show appliance powers each in a single row of subplots.
    ###
    # Find max value in predictions for setting plot limits.
    max_pred = np.ceil(np.max(list(predicted_powers.values())))
    nrows = len(args.appliances) + 1
    fig, ax = plt.subplots(nrows=nrows, ncols=1, constrained_layout=True)
    ax[0].set_ylabel('Watts')
    ax[0].set_title('aggregate')
    ax[0].plot(aggregate[offset:-offset], color='orange', linewidth=1.8)
    # Set friendly looking date and times on x-axis.
    locator = AutoDateLocator()
    ax[0].xaxis.set_major_locator(locator=locator)
    ax[0].xaxis.set_major_formatter(AutoDateFormatter(locator=locator))
    fig.autofmt_xdate()
    # Predicted dataset size is used to crop input dataset since test_provider
    # defaults to outputting complete batches.
    # See `allow_partial_batches' parameter in window_generator.
    predicted_dataset_size = predicted_powers[DEFAULT_APPLIANCES[0]].size
    row = 1
    for appliance in args.appliances:
        ax[row].set_ylabel('Watts')
        ax[row].set_title(appliance)
        ax[row].set_ylim(0, max_pred)
        ax[row].plot(
            mains_gt_df['datetime'][offset:-offset][:predicted_dataset_size],
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
                mains_gt_df['datetime'], mains_gt_df[appliance],
                label='ground_truth', color='blue'
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
        mains_gt_df['datetime'][offset:-offset], aggregate[offset:-offset],
        color=color_names[0], linewidth=1.8, linestyle=line_styles[0],
        label='Aggregate'
    )
    # Set friendly looking date and times on x-axis.
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator=locator)
    ax.xaxis.set_major_formatter(AutoDateFormatter(locator=locator))
    fig.autofmt_xdate()
    # Plot appliance powers.
    for i, appliance in enumerate(args.appliances):
        ax.plot(
            mains_gt_df['datetime'][offset:-offset][:predicted_dataset_size],
            predicted_powers[appliance], color=color_names[i+1],
            linewidth=1.5, linestyle = line_styles[i+1], label=appliance
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

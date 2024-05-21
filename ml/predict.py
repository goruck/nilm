"""
Predict and analyze appliance type and power using novel data from my home.

Actual mains power is captured in a csv file by real-time telemetry code
running on a Raspberry Pi that also performs load disaggregation inference
using TFLite models converted from the trained TensorFlow floating point
models. These TensorFlow models are used here to also perform load disaggregation
and the results can be compared against the TFLite inferences. Ground truth power
consumption for the appliances are captured by another telemetry program
which can be used to evaluate both the TensorFlow and TFLite inference results.

See rpi/infer.py for real-time mains power telemetry and inference code.

See https://github.com/goruck/simple-energy-logger real-time ground truth telemetry code.

Copyright (c) 2022~2024 Lindo St. Angel
"""

import os
import argparse
import socket
import glob
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from matplotlib import rcParams
import pandas as pd

import common
from logger import Logger
from window_generator import WindowGenerator
from nilm_metric import NILMTestMetrics

# Inference input sample window length for all appliances
WINDOW_LENGTH = 599 # units: samples

# Center of window where inference result is placed.
WINDOW_CENTER = int(0.5 * (WINDOW_LENGTH - 1.0)) # units: samples

# Names of all appliances that can be monitored.
DEFAULT_APPLIANCES = ['kettle', 'fridge', 'microwave', 'washingmachine', 'dishwasher']

# csv file name prefix of mains power and appliance predictions.
# These files are in the 'default_dataset_dir' identified below.
RT_DATA_FILE_PREFIX = 'samples'

# csv file name of ground truth appliance power readings.
# These files are in the 'default_dataset_dir' identified below.
GT_DATA_FILE_PREFIX = 'appliance_energy_data_'

# Set local time zone. Dataset datetimes are UTC.
TZ = 'America/Los_Angeles'

def get_arguments():
    """Get command line arguments."""
    default_trained_model_dir = '/home/lindo/Develop/nilm/ml/models'
    default_results_dir = '/home/lindo/Develop/nilm/ml/results'
    default_dataset_dir = '/home/lindo/Develop/nilm-datasets/my-house'
    default_panel_location = 'house'
    default_not_before = '1970-01-01 00:00:00'

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
        '--not_before',
        type=datetime.fromisoformat,
        default=default_not_before,
        help='do not use data before this date (ISO 8601 format)'
    )
    parser.add_argument(
        '--model_arch',
        type=str,
        default=['cnn', 'cnn_fine_tune', 'transformer'],
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

def get_real_power(
        file_name:str,
        not_before:datetime,
        zone:str=TZ
    ) -> pd.DataFrame:
    """Load real-time dataset and return total real power with datetimes."""
    dataframe = pd.read_csv(
        file_name,
        index_col=0,
        usecols=['DT', 'W1', 'W2'],
        parse_dates=['DT'],
        date_format='ISO8601'
    )
    # Localize datetimes.
    dataframe = dataframe.tz_convert(tz=zone)
    # Filter datetimes earlier than threshold.
    dataframe = dataframe[dataframe.index >= not_before.isoformat()]
    # Compute total real power and insert into dataframe.
    # W1 and W2 are the real power in each phase.
    real_power = dataframe['W1'] + dataframe['W2']
    dataframe.insert(loc=0, column='real_power', value=real_power)
    # Remove W1 and W2 since they are no longer needed.
    dataframe = dataframe.drop(columns=['W1', 'W2'])
    return dataframe.fillna(0)

def get_ground_truth(
        file_name:str,
        not_before:datetime,
        zone:str=TZ
    ) -> pd.DataFrame:
    """Load ground truth dataset and return appliance active power with datetimes.

    This reads a ground truth csv file as a dataframe, localizes it, converts it
    from a row format to column format, corrects for inconsistent appliance naming,
    adds missing default appliances and downsamples to match mains sample rate.

    If the dataset has fewer appliances than what is defined in DEFAULT_APPLIANCES,
    the missing appliance(s) will be added with all power values set to NaN.

    Args:
        file_name: Full path to ground truth dataset in csv format with datetimes.
        not_before: ISO 8601 datetime where data before this will not be returned.
        zone: Localize datetimes in dataset to this time zone.

    Returns:
        Dataframe of appliance ground truth power with datetime index.
    """
    dataframe = pd.read_csv(
        file_name,
        index_col=0,
        usecols=['date', 'appliance', 'apower'],
        parse_dates=['date'],
        date_format='ISO8601'
    )

    # Localize datetimes.
    dataframe = dataframe.tz_convert(tz=zone)

    # Filter datetimes earlier than threshold.
    dataframe = dataframe[dataframe.index >= not_before.isoformat()]

    # Row to column format transformation.
    def row_to_column(x:str) -> pd.DataFrame:
        """Convert rows of appliance data to columns."""
        force_consistent_names = 'washing machine' if x == 'washingmachine' else x
        df = dataframe.loc[dataframe['appliance'] == force_consistent_names]
        df = df.drop(columns=['appliance'])
        return df.rename(columns={'apower': x})
    all_columns = [row_to_column(a) for a in DEFAULT_APPLIANCES]
    dataframe = pd.concat(all_columns).sort_index()

    # Downsample to match global sample rate.
    # Although the appliance ground truth power was sampled at 1 data point every 8 seconds,
    # each appliance was read consecutively by the logging program with about 1 second
    # between reads. Therefore the dataset has groups of appliance readings clustered
    # around 8 second intervals that this downsampling corrects for.
    resample_period = f'{common.SAMPLE_PERIOD}s'
    return dataframe.resample(resample_period, origin='start').max()

def get_realtime_predictions(
        file_name:str,
        not_before:datetime,
        zone=TZ
    ) -> pd.DataFrame:
    """Load appliance power predictions that were preformed in real-time with datetimes."""
    # Load real-time prediction dataset.
    dataframe = pd.read_csv(
        file_name,
        index_col=0, # make datetime the index
        usecols=['DT']+DEFAULT_APPLIANCES,
        parse_dates=['DT'],
        date_format='ISO8601'
    )
    # Localize datetimes.
    dataframe = dataframe.tz_convert(tz=zone)
    # Filter datetimes earlier than threshold.
    dataframe = dataframe[dataframe.index >= not_before.isoformat()]
    # Adjustment for real-time prediction timing.
    # Adjustment is done by moving samples earlier in time by a value equal
    # to the window center since the real-time code places the prediction
    # at the end of a window of samples.
    dataframe = dataframe.shift(periods=-WINDOW_CENTER, fill_value=0)
    return dataframe.fillna(0)

def compute_metrics(
        ground_truth:pd.DataFrame,
        ground_truth_status:np.ndarray,
        pred:pd.DataFrame,
        pred_status:np.ndarray
    ) -> str:
    """Assess performance of prediction vs ground truth.

    Args:
        ground_truth: Appliance ground truth power.
        ground_truth_status: Appliance ground truth on-off status.
        pred: Appliance predicted power.
        pred_status: Appliance predicted on-off status.

    Returns:
        String containing appliance performance metrics vs ground truth.

    Raises:
        ValueError if there are fewer predictions than ground truth samples.
    """
    gt_len, p_len = len(ground_truth.index), len(pred.index)

    # There should never be more predictions than ground truth samples because
    # the latter is based on the number of real power samples which are used to
    # make predictions. There can be, however, NaNs in the ground truth samples.
    if p_len > gt_len:
        raise ValueError('Cannot have more predictions than ground truth samples.')

    # Adjust ground truth sample size to match number of predictions.
    # This is to cover the case where the number of predictions are less
    # than the number of ground truth samples because by default partial
    # batch sizes are not allowed when making predictions.
    if gt_len > p_len:
        ground_truth.drop(ground_truth.tail(gt_len - p_len).index, inplace=True)
        ground_truth_status = ground_truth_status[:p_len]

    # Filter out NaN rows in datasets based on NaN values in the ground truth data.
    # NaN values can arise from appliances that were not monitored or drop-outs.
    gt_not_na_rows = np.array(ground_truth.notna())
    ground_truth = ground_truth[gt_not_na_rows]
    ground_truth_status = ground_truth_status[gt_not_na_rows]
    pred = pred[gt_not_na_rows]
    pred_status = pred_status[gt_not_na_rows]

    metrics = NILMTestMetrics(
        target=np.array(ground_truth),
        target_status=ground_truth_status,
        prediction=np.array(pred),
        prediction_status=pred_status,
        sample_period=common.SAMPLE_PERIOD
    )

    epd_gt = metrics.get_epd(ground_truth * ground_truth_status, common.SAMPLE_PERIOD)
    epd_pred = metrics.get_epd(pred * pred_status, common.SAMPLE_PERIOD)

    return (
        f'True positives: {metrics.get_tp()}\n'
        f'True negatives: {metrics.get_tn()}\n'
        f'False positives: {metrics.get_fp()}\n'
        f'False negatives: {metrics.get_fn()}\n'
        f'Accuracy: {metrics.get_accuracy()}\n'
        f'MCC: {metrics.get_mcc()}\n'
        f'F1: {metrics.get_f1()}\n'
        f'MAE: {metrics.get_abs_error()["mean"]} (W)\n'
        f'NDE: {metrics.get_nde()}\n'
        f'SAE: {metrics.get_sae()}\n'
        f'Ground truth EPD: {epd_gt} (Wh)\n'
        f'Predicted EPD: {epd_pred} (Wh)\n'
        f'EPD Relative Error: {100.0 * (epd_pred - epd_gt) / epd_gt} (%)'
    )

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
    rp_df_from_each_file = (
        get_real_power(f, not_before=args.not_before) for f in all_rt_files
    )
    rp_df = pd.concat(rp_df_from_each_file)
    rp_df.sort_index(inplace=True)
    logger.log(f'There are {rp_df.size/10**6:.3f}M real power samples.')

    # Get all ground truth datasets.
    all_gt_files = glob.glob(
        os.path.join(args.datadir, args.panel) + f'/{GT_DATA_FILE_PREFIX}*.csv'
    )
    logger.log(f'Found ground truth files: {all_gt_files}.')
    gt_df_from_each_file = (
        get_ground_truth(f, not_before=args.not_before) for f in all_gt_files
    )
    gt_df = pd.concat(gt_df_from_each_file)
    gt_df.sort_index(inplace=True)

    # Merge mains dataset with ground truth dataset.
    # Since mains was sampled asynchronously wrt ground truth, use asof merge.
    # This will align ground truth datetimes with real power datetimes.
    gt_df = pd.merge_asof(rp_df, gt_df, left_index=True, right_index=True, direction='nearest')

    # Compute ground truth status.
    gt_status = np.array(
        [common.compute_status(np.array(gt_df.loc[:,a]), a) for a in args.appliances]
    ).transpose()

    # Get the mains (aka aggregate) real power and perform inference with it.
    aggregate = np.array(rp_df, dtype=np.float32)
    logger.log('Making power predictions.')
    predicted_powers = {
        appliance: predict(
            appliance, aggregate, args.model_dir, args.model_arch
        ) for appliance in args.appliances
    }

    # Create a dataframe containing the predicted power.
    # Predicted dataset size is used to crop dataset since test_provider
    # defaults to outputting complete batches, so there will be more mains
    # samples than predictions. See `allow_partial_batches' in window_generator.
    predicted_dataset_size = predicted_powers[args.appliances[0]].size
    # Add back datetime index and adjustment for prediction timing.
    # Adjustment is done by moving samples later in time by a value equal
    # to the window center since the predict function places the prediction
    # at the beginning of a window of samples.
    pp_df = pd.DataFrame(predicted_powers)
    pp_df.set_index(rp_df.index[:predicted_dataset_size], inplace=True)
    pp_df = pp_df.shift(periods=WINDOW_CENTER, fill_value=0)

    # Compute and apply status to predictions.
    pp_status = np.array(
        [common.compute_status(np.array(pp_df.loc[:,a]), a) for a in args.appliances]
    ).transpose()
    pp_df *= pp_status

    # Get real-time prediction dataset and apply status.
    rt_df_from_each_file = (
        get_realtime_predictions(f, not_before=args.not_before) for f in all_rt_files
    )
    rt_df = pd.concat(rt_df_from_each_file)
    rt_df.sort_index(inplace=True)
    rt_df = rt_df[args.appliances] # cover case where subset of appliances are specified
    rt_status = np.array(
        [common.compute_status(np.array(rt_df.loc[:,a]), a) for a in args.appliances]
    ).transpose()
    rt_df *= rt_status

    # Compute metrics for predictions.
    logger.log('\n***Performance metrics of predictions vs ground truth***')
    for i, a in enumerate(args.appliances):
        logger.log(f'\nPredictions vs Ground Truth Metrics for {a}:')
        result = compute_metrics(
            ground_truth=gt_df.loc[:,a],
            ground_truth_status=gt_status[:,i],
            pred=pp_df.loc[:,a],
            pred_status=pp_status[:,i]
        )
        logger.log(result)
    logger.log('\n***Performance metrics of real-time predictions vs ground truth***')
    for i, a in enumerate(args.appliances):
        logger.log(f'\nRT Predictions vs Ground Truth Metrics for {a}:')
        result = compute_metrics(
            ground_truth=gt_df.loc[:,a],
            ground_truth_status=gt_status[:,i],
            pred=rt_df.loc[:,a],
            pred_status=rt_status[:,i]
        )
        logger.log(result)

    # Plot setup
    rcParams['timezone'] = TZ
    locator = AutoDateLocator(tz=TZ)

    ###
    ### Save and show appliance powers each in a single row of subplots.
    ###
    # Find max value in predictions for setting plot limits.
    max_pred = np.ceil(np.max(list(predicted_powers.values())))
    nrows = len(args.appliances) + 1
    fig, ax = plt.subplots(nrows=nrows, ncols=1)
    ax[0].set_ylabel('Watts')
    # Plot mains power.
    ax[0].set_title('aggregate')
    ax[0].plot(rp_df, color='orange')
    # Set friendly looking date and times on x-axis.
    ax[0].xaxis.set_major_locator(locator=locator)
    ax[0].xaxis.set_major_formatter(AutoDateFormatter(locator=locator))
    fig.autofmt_xdate()
    for row, appliance in enumerate(args.appliances, start=1):
        ax[row].set_ylabel('Watts')
        ax[row].set_title(appliance)
        ax[row].set_ylim(0, max_pred)
        ax[row].plot(pp_df[appliance], color='red', label='prediction')
        if args.show_rt_preds:
            ax[row].plot(rt_df[appliance], color='green', label='real-time prediction')
            ax[row].legend(loc='upper right')
        if args.show_gt:
            ax[row].plot(gt_df[appliance], color='blue', label='ground_truth')
            ax[row].legend(loc='upper right')
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
    fig, ax = plt.subplots()
    fig.suptitle(f'Prediction results for {args.panel}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Watts')
    ax.set_xlabel('Date-Time')
    # Plot mains power.
    ax.plot(rp_df, label='aggregate')
    # Set friendly looking date and times on x-axis.
    ax.xaxis.set_major_locator(locator=locator)
    ax.xaxis.set_major_formatter(AutoDateFormatter(locator=locator))
    fig.autofmt_xdate()
    # Plot appliance powers.
    ax.plot(pp_df, label=pp_df.columns.to_list())
    ax.legend(loc='upper right')
    # Save and optionally show plot.
    plot_savepath = os.path.join(
        args.results_dir, args.panel, f'predict_{args.model_arch}_plot.png'
    )
    plt.savefig(fname=plot_savepath)
    if args.plot:
        plt.show()
    plt.close()

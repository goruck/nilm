"""
Create train, test and validation datasets from my house's ground truth data.

Actual mains power is captured in a csv file by real-time telemetry code
running on a Raspberry Pi (that also performs load disaggregation inference
using TFLite models converted from the trained TensorFlow floating point
models, but is not used here). Ground truth power consumption for the appliances
are captured by another telemetry program which is aligned here with mains power,
pre-processed and saved as a csv file for downstream ML training and testing.

See rpi/infer.py for real-time mains power telemetry and inference code.

See https://github.com/goruck/simple-energy-logger real-time ground truth telemetry code.

Copyright (c) 2024 Lindo St. Angel
"""

import argparse
import socket
import glob
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import pandas as pd

sys.path.append('../../')
import common
from logger import Logger

# Names of all appliances that can be monitored.
DEFAULT_APPLIANCES = ['kettle', 'fridge', 'microwave', 'washingmachine', 'dishwasher']

# csv file name prefix of mains power and appliance predictions.
# These files are in the 'default_dataset_dir' identified below.
RT_DATA_FILE_PREFIX = 'samples'

# csv file name of ground truth appliance power readings.
# These files are in the 'default_dataset_dir' identified below.
GT_DATA_FILE_PREFIX = 'appliance_energy_data_'

# Define fractions of source dataset to create train, val and test datasets.
# Must add up to 1.0.
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.2
TEST_FRACTION = 0.0

def get_arguments():
    """Get command line arguments."""
    default_results_dir = '/home/lindo/Develop/nilm/ml/dataset_management/my-house'
    default_dataset_dir = '/home/lindo/Develop/nilm-datasets/my-house'
    default_panel_location = 'house'

    parser = argparse.ArgumentParser(
        description='Create ground truth datasets using novel data from my home.'
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
        help='novel source dataset(s) directory'
    )
    parser.add_argument(
        '--panel',
        type=str,
        default=default_panel_location,
        help='sub-panel location'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=default_results_dir,
        help='directory to save the generated datasets'
    )
    parser.add_argument(
        '--crop',
        type=int,
        default=None,
        help='use part of the dataset for generation'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='if set, show datasets for visually inspection'
    )
    parser.set_defaults(plot=False)

    return parser.parse_args()

def get_real_power(file_name, crop=None, zone='America/Los_Angeles') -> pd.DataFrame:
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
    # Compute total real power and insert into dataframe.
    # W1 and W2 are the real power in each phase.
    real_power = dataframe['W1'] + dataframe['W2']
    dataframe.insert(loc=0, column='aggregate', value=real_power)
    # Remove W1 and W2 since they are no longer needed.
    dataframe = dataframe.drop(columns=['W1', 'W2'])
    return dataframe.fillna(0)

def get_ground_truth(file_name, crop=None, zone='America/Los_Angeles') -> pd.DataFrame:
    """Load ground truth dataset and return appliance active power with datetimes.

    This reads a ground truth csv file, as a dataframe, localizes it, converts it
    from a row format to column format, corrects for inconsistent appliance naming,
    adds missing default appliances and downsamples to match mains sample rate.

    If the dataset has fewer appliances than what is defined in DEFAULT_APPLIANCES,
    the missing appliance(s) will be added with all power values set to NaN.

    Args:
        file_name: Full path to ground truth dataset in csv format with datetimes.
        crop: Get only this number of rows from dataset.
        zone: Localize datetimes in dataset to this time zone.

    Returns:
        Dataframe of appliance ground truth power with datetime index.
    """
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
    def row_to_column(x:str) -> pd.DataFrame:
        """Convert rows of appliance data to columns."""
        force_consistent_names = 'washing machine' if x == 'washingmachine' else x
        d = dataframe.loc[dataframe['appliance'] == force_consistent_names]
        d = d.drop(columns=['appliance'])
        return d.rename(columns={'apower': x})
    all_columns = [row_to_column(a) for a in DEFAULT_APPLIANCES]
    dataframe = pd.concat(all_columns).sort_index()
    # Downsample to match global sample rate.
    # Although the appliance ground truth power was sampled at 1 data point every 8 seconds,
    # each appliance was read consecutively by the logging program with about 1 second
    # between reads. Therefore the dataset has groups of appliance readings clustered
    # around 8 second intervals that this downsampling corrects for.
    resample_period = f'{common.SAMPLE_PERIOD}s'
    return dataframe.resample(resample_period, origin='start').max()

if __name__ == '__main__':
    args = get_arguments()

    logger = Logger(
        log_file_name=Path(
            args.results_dir, f'create_ground_truth_datasets_{args.panel}.log'
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
    all_rt_files = glob.glob(f'{args.datadir}/{args.panel}/{RT_DATA_FILE_PREFIX}*.csv')
    logger.log(f'Found mains real power files: {all_rt_files}.')
    rp_df_from_each_file = (get_real_power(f, crop=args.crop) for f in all_rt_files)
    rp_df = pd.concat(rp_df_from_each_file)
    rp_df.sort_index(inplace=True)

    # Get all ground truth datasets.
    all_gt_files = glob.glob(f'{args.datadir}/{args.panel}/{GT_DATA_FILE_PREFIX}*.csv')
    logger.log(f'Found ground truth files: {all_gt_files}.')
    gt_df_from_each_file = (get_ground_truth(f, crop=args.crop) for f in all_gt_files)
    gt_df = pd.concat(gt_df_from_each_file)
    gt_df.sort_index(inplace=True)

    # Merge mains dataset with ground truth dataset.
    # Since mains was sampled asynchronously wrt ground truth, use asof merge.
    # This will align ground truth datetimes with real power samples.
    # Ground truth datetimes without corresponding mains datetimes will be dropped.
    gt_df = pd.merge_asof(rp_df, gt_df, left_index=True, right_index=True, direction='nearest')

    gt_df_num_rows = gt_df.shape[0]
    logger.log(f'Number of ground truth samples per appliance: {gt_df_num_rows}.')

    num_train = int(gt_df_num_rows * TRAIN_FRACTION)
    logger.log(f'Number of training samples: {num_train}')
    num_val = int(gt_df_num_rows * VAL_FRACTION)
    logger.log(f'Number of validation samples: {num_val}')
    num_test = int(gt_df_num_rows * TEST_FRACTION)
    logger.log(f'Number of test samples: {num_test}')
    if num_train + num_val + num_test > gt_df_num_rows:
        raise RuntimeError('Number of dataset samples exceeds number in source.')

    train_df = gt_df[:num_train]
    val_df = gt_df[num_train: num_train + num_val]
    test_df = gt_df[num_train + num_val: num_train + num_val + num_test]
    datasets = {'training': train_df, 'validation': val_df, 'test': test_df}

    for k, v in datasets.items():
        logger.log(f'Working on dataset: {k}.')

        if v.empty:
            logger.log('Empty dataset, skipping.')
            continue

        for appliance in args.appliances:
            logger.log(f'Working on appliance: {appliance}.')

            # Select aggregate and appliance power.
            df = v[['aggregate', appliance]]

            # Drop NaN's.
            df_appliance_nan = df[appliance].notna() #row is False if NaN, else True
            logger.log(f'Number of NaN rows dropped: {(~df_appliance_nan).values.sum()}')
            df = df[df_appliance_nan]

            # Clamp power.
            max_on_power = common.params_appliance[appliance]['max_on_power']
            logger.log(f'Clamping appliance power to: {max_on_power} (W)')
            df.loc[:, appliance] = df.loc[:, appliance].clip(0, max_on_power)

            # Get appliance status and add to end of dataframe.
            logger.log('Computing on-off status.')
            status = common.compute_status(df.loc[:, appliance].to_numpy(), appliance)
            df.insert(2, 'status', status)
            num_on = len(df[df['status']==1])
            num_off = len(df[df['status']==0])
            num_total = df.iloc[:, 2].size
            logger.log(f'Number of samples with on status: {num_on}.')
            logger.log(f'Number of samples with off status: {num_off}.')
            logger.log(f'Number of total samples: {num_total}.')
            if num_on + num_off != num_total:
                assert RuntimeError('Total of on and off activations must equal number of samples.')

            if args.plot:
                # Plot aggregate power, appliance power and status for visual inspection.
                fig, ax = plt.subplots()
                fig.suptitle(
                    f'Ground truth {k} dataset for {args.panel}', fontsize=16, fontweight='bold'
                )
                ax.set_ylabel('Watts')
                ax.set_xlabel('DateTime')
                # Plot mains power.
                ax.plot(df['aggregate'], label='aggregate', color='blue')
                # Set friendly looking date and times on x-axis.
                locator = AutoDateLocator()
                ax.xaxis.set_major_locator(locator=locator)
                ax.xaxis.set_major_formatter(AutoDateFormatter(locator=locator))
                fig.autofmt_xdate()
                # Plot appliance power.
                ax.plot(df[appliance], label=appliance, color='red')
                ax.legend(loc='upper right', fontsize='x-small')
                # Plot status on right side.
                ax2 = ax.twinx()
                ax2.set_ylabel('status', color='green')
                ax2.plot(df['status'], label='status', color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                plt.show()
                plt.close()

            # Standardize aggregate dataset.
            if common.USE_ALT_STANDARDIZATION:
                agg_mean = common.ALT_AGGREGATE_MEAN
                agg_std = common.ALT_AGGREGATE_STD
            else:
                agg_mean = common.params_appliance[appliance]['train_agg_mean']
                agg_std = common.params_appliance[appliance]['train_agg_std']
            logger.log(f'Standardizing aggregate with mean {agg_mean} (W) and std {agg_std} (W).')
            df.loc[:, 'aggregate'] = (df.loc[:, 'aggregate'] - agg_mean) / agg_std

            # Scale appliance dataset.
            if common.USE_APPLIANCE_NORMALIZATION:
                # Normalize appliance dataset to [0, max_on_power].
                app_min = 0.0
                app_max = max_on_power
                logger.log(f'Normalizing appliance with min {app_min} (W) and max {app_max} (W).')
                df.loc[:, appliance] = (df.loc[:, appliance] - app_min) / (app_max - app_min)
            else:
                # Standardize appliance dataset.
                if common.USE_ALT_STANDARDIZATION:
                    app_mean = common.params_appliance[appliance]['alt_app_mean']
                    app_std = common.params_appliance[appliance]['alt_app_std']
                else:
                    app_mean = common.params_appliance[appliance]['train_app_mean']
                    app_std = common.params_appliance[appliance]['train_app_std']
                logger.log(
                    'Using alt standardization.' if common.USE_ALT_STANDARDIZATION 
                    else 'Using default standardization.'
                )
                logger.log(
                    f'Standardizing appliance with mean {app_mean} (W) and std {app_std} (W).'
                )
                df.loc[:, appliance] = (df.loc[:, appliance] - app_mean) / app_std

            # Save dataset as a csv file.
            file_path = f'{args.results_dir}/{appliance}'
            Path(file_path).mkdir(exist_ok=True)
            csv_file_name = f'{appliance}_{k}_{args.panel}.csv'
            logger.log(f'Saving dataset as {csv_file_name}.')
            df.to_csv(Path(file_path, csv_file_name), index=False)

    logger.log('Successfully completed.')

"""Class to computes NILM test metrics.

Copyright (c) 2022~2023 Lindo St. Angel.
"""

import numpy as np

class NILMTestMetrics():
    """Computes NILM-related test metrics.

    Provides various metrics useful for evaluating predictions from NILM models.
    Data inputs are ground truth and predicted appliance power samples in Watts.
    The sample update period must be provided for energy calculations and
    an appliance on-off threshold must be provided to determine status.

    Typical usage example:
    ```python
    predicted_energy_per_day = NILMTestMetrics.get_epd(prediction, SAMPLE_PERIOD)
    metrics = NILMTestMetrics(
        target=ground_truth, prediction=prediction,
        threshold=threshold, sample_period=SAMPLE_PERIOD)
    f1 = metrics.get_f1()
    ```

    Attributes:
        None
    """
    def __init__(
        self,
        target:np.ndarray,
        target_status:np.ndarray,
        prediction:np.ndarray,
        prediction_status:np.ndarray,
        sample_period:int
    ) -> None:
        """Initializes NILM metrics class.
        
        Args:
            target: Ground truth power consumption samples in Watts.
            target_status: Ground truth on-off status.
            prediction: Predicted power consumption samples in Watts.
            prediction_status: Prediction on-off status.
            sample_period: Sample update period in seconds.

        Raises:
            ValueError if target and prediction datasets are not same size.
        """
        if target.shape != prediction.shape:
            raise ValueError(
                f'Target {target.shape} and prediction {prediction.shape} must be same shape.'
            )
        if target.shape != target_status.shape:
            raise ValueError(
                f'Target shape {target.shape} must match target status shape {target_status.shape}.'
            )
        if prediction.shape != prediction_status.shape:
            raise ValueError(
                f'Prediction shape {prediction.shape} must match '
                f'prediction status shape {prediction_status.shape}.'
            )

        self.target = target
        self.target_status = target_status
        self.prediction = prediction
        self.prediction_status = prediction_status
        self.sample_period = sample_period

    def _get_statistics(self, data:np.ndarray) -> dict:
        """Generate general statistics of a dataset."""
        mean = np.mean(data)
        std = np.std(data)
        min_v = np.sort(data)[0]
        max_v = np.sort(data)[-1]

        quartile1 = np.percentile(data, 25)
        median = np.percentile(data, 50)
        quartile2 = np.percentile(data, 75)

        return {
            'mean':mean,
            'std':std,
            'min':min_v,
            'max':max_v,
            'quartile1':quartile1,
            'median':median,
            'quartile2':quartile2
        }

    def get_tp(self) -> int:
        '''Returns the number of true positives between target and prediction.

        True positives are when both target and predicted values are True.
        '''
        tp_array = np.logical_and(self.target_status, self.prediction_status)
        return np.sum(tp_array.astype('int'))

    def get_fp(self) -> int:
        '''Returns the number of false positives between target and prediction.

        False positives are when target values are False and predicted values are True.
        '''
        # Note: (1 - status) inverts the status.
        fp_array = np.logical_and(1-self.target_status, self.prediction_status)
        return np.sum(fp_array.astype('int'))

    def get_fn(self) -> int:
        '''Returns the number of false negatives between target and prediction.

        False negatives are when target values are True and predicted values are False.
        '''
        # Note: (1 - status) inverts the status.
        fn_array = np.logical_and(self.target_status, 1-self.prediction_status)
        return np.sum(fn_array.astype('int'))

    def get_tn(self) -> int:
        '''Returns the number of true negatives between target and prediction.

        True negatives are when the target and predicted values are both False.
        '''
        # Note: (1 - status) inverts the status.
        tn_array = np.logical_and(1-self.target_status, 1-self.prediction_status)
        return np.sum(tn_array.astype('int'))

    def get_mcc(self) -> float:
        """Returns the Matthews Correlation Coefficient."""
        tp = float(self.get_tp())
        tn = float(self.get_tn())
        fp = float(self.get_fp())
        fn = float(self.get_fn())

        return (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    def get_recall(self) -> float:
        '''Returns the recall rate.'''
        tp = float(self.get_tp())
        fn = float(self.get_fn())

        if tp + fn <= 0.0:
            recall = tp / (tp + fn + 1e-9)
        else:
            recall = tp / (tp + fn)
        return recall

    def get_precision(self) -> float:
        '''Returns the precision rate.'''
        tp = float(self.get_tp())
        fp = float(self.get_fp())

        if tp + fp <= 0.0:
            precision = tp / (tp + fp + 1e-9)
        else:
            precision = tp / (tp + fp)
        return precision

    def get_f1(self) -> float:
        '''Returns the F1 score.'''
        recall = self.get_recall()
        precision = self.get_precision()

        if precision == 0.0 or recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        return f1

    def get_accuracy(self) -> float:
        '''Returns the accuracy rate.'''
        tp = float(self.get_tp())
        tn = float(self.get_tn())

        return (tp + tn) / self.target.size

    def get_relative_error(self) -> float:
        '''Returns the relative error.'''
        return np.mean(
            np.nan_to_num(
                np.abs(
                    self.target * self.target_status - self.prediction * self.prediction_status
                ) / self.target.size
            )
        )

    def get_abs_error(self) -> dict:
        """Returns absolute error statistics.

        Evaluates the absolute difference between the prediction and the
        ground truth at every time point and returns assoc. statistics.
        """
        data = np.abs(self.target * self.target_status - self.prediction * self.prediction_status)

        return self._get_statistics(data)

    def get_nde(self) -> float:
        '''Returns the normalized disaggregation error (nde).

        Evaluates the normalized error of the squared difference
        between the prediction and the ground truth.
        '''
        return np.sum(
            ( self.target * self.target_status - self.prediction * self.prediction_status)**2
        ) / np.sum((self.target**2))

    def get_sae(self) -> float:
        '''Returns the signal aggregate error (sae).

        Evaluates the relative error of the total energy consumption
        between the prediction and ground truth.

        sae = |rhat-r| / r
        where r is the ground truth total energy and
        rhat is the predicted total energy.
        '''
        # data[i] * sample_period / (60 * 60) is instantaneous energy in Wh.
        r = np.sum(self.target * self.target_status * float(self.sample_period) / 3600.0)
        rhat = np.sum(self.prediction * self.prediction_status * float(self.sample_period) / 3600.0)

        return np.abs(r - rhat) / r

    @staticmethod
    def get_epd(data:np.ndarray, sample_period:int) -> float:
        '''Returns NILM dataset average energy per day in Watt-hours.

        Args:
            data: power consumption samples in Watts, updated every sample_period.
            sample_period: sample update period in seconds.
        '''
        # Calculate number of samples per hour in dataset.
        sph = 60 * 60 // sample_period

        # Calculate total samples in dataset.
        if data.size - sph < 1:
            raise ValueError('Need at least one hour of samples to calculate epd.')
        tps = data.size - sph

        # Generate a list of power consumed for each hour in dataset.
        # data[i:i+sph] is one hour of power consumption samples,
        # summing those samples and dividing by number of samples
        # gives energy in Watt-hours for that interval.
        watt_hours = [np.sum(data[i:i+sph]) / sph for i in range(0, tps, sph)]

        # Calculate number of days in dataset.
        days = data.size / sph / 24

        # Return average energy per day in Watt-hour for entire dataset.
        return np.sum(watt_hours) / days

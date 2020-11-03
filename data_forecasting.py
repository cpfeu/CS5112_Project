'''

ARIMA
SARIMA
Moving Average Models
ETS Models
Vector Autoregression
Support Vector Autoregression

'''
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from collections import defaultdict

class SimpleExponentialSmoothing:
    def __init__(self, parser_object):
        self.parser_object = parser_object
        self.ALPHA_MIN = 0.01
        self.ALPHA_MAX = 0.999
        self.ALPHA_STEP = 0.01
        self.train_size, self.test_size, self.series, self.train_test_size, self.horizon = None, None, None, None, None

    def get_rmse(self, a, b):
        """
        Return scalar of RMSE between two lists
        """
        return math.sqrt(np.mean((np.array(a) - np.array(b)) ** 2))

    def simple_single_exponential_smoothing(self, series, horizon, alpha=0.5):
        """
        Return a series of smooth points given a series, alpha constant, and forecast horizon
        Calculates forecast from weighted averages, where the weights decrease exponentially as observations come
            further from the past, a rate at which is controlled by the alpha parameter. If alpha is large, more weight
            is given to recent observations, if alpha is small, more weight is given to historical observations
        Inputs
            series:     series for forecast
            horizon:    forecast horizon
            alpha:      smoothing constant
                        When alpha closer to 0, slow dampening
                        When alpha is closer to 1, quick dampening
        Outputs
            result:     Forecast predictions of length horizon
        """
        result = [0, series[0]]
        for i in range(1, len(series) + horizon - 1):
            if i >= len(series):
                result.append((series[-1] * alpha) + ((1-alpha) * result[i]))
            else:
                result.append((series[i] * alpha) + ((1-alpha) * result[i]))
        return result[len(series):len(series)+horizon]

    def simple_double_exponential_smoothing(self, series, horizon, alpha=0.3, beta=0.3):
        """
        Return a series of smooth points given a series, alpha constant, beta constant, and forecast horizon
        Double exponential smoothing adapts single exponential smoothing but adds in addition to the level equation a trend equation
            the trend equation estimates a trend of the series at time. Thus the horizon forecast is equal to the last estimated
            level plus horizon times the last estimated trend value
        This is also known as Holt's linear trend method
        Inputs
            series:     series for forecast
            horizon:    forecast horizon
            alpha:      smoothing constant
                        When alpha closer to 0, slow dampening
                        When alpha is closer to 1, quick dampening
            beta:       trend smoothing constant
        Outputs
            result:     Forecast predictions of length horizon
        """
        result = [0, series[0]]
        level, trend = series[0], series[1] - series[0]
        for i in range(1, len(series) + horizon - 1):
            if i >= len(series):
                m = i - len(series) + 2
                result.append(level + m * trend)
            else:
                value = series[i]
                last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
                trend = beta * (level - last_level) + (1 - beta) * trend
                result.append(level + trend)
        return result[len(series):len(series) + horizon]

    def initialize_model(self, horizon=7, train_test_split=0.8):
        time_stamp_list = []
        close_list = []
        for single_google_recording in self.parser_object.single_google_recording_list:
            time_stamp_list.append(single_google_recording.time_stamp)
            close_list.append(single_google_recording.close)

        print(datetime.now(), ': Exponential Smoothing Model Received Data')
        self.horizon = horizon
        self.series = close_list
        self.train_size = int((len(close_list)-horizon-1) * train_test_split)
        self.test_size = len(close_list)-horizon-1-self.train_size
        self.train_test_size = self.train_size + self.test_size

    def get_error_metrics(self, series, train_size, horizon, alpha):
        """
        Return predictions and errors for series of a train validation split and alpha parameter
        Inputs
            series     : series to forecast, with length = (train_size + val_size)
            train_size : length of series to use as train ie. train set is series[:train_size]
            horizon    : forecast horizon
        Outputs
            mean rmse
        """
        rmse = []
        preds_dict = {}
        for i in range(train_size, len(series) - horizon + 1, int(horizon / 2)):
            preds_list = self.simple_single_exponential_smoothing(series[i - train_size:i], horizon, alpha)
            rmse.append(self.get_rmse(series[i:i + horizon], preds_list))
            preds_dict[i] = preds_list

        return np.mean(rmse), preds_dict

    def hyperparam_tune_alpha(self, series, train_size, horizon):
        """
        Given a series, tune hyperparameter alpha, fit and predict
        Inputs
            series     : series to forecast
            train_size : length of series to use train set
            horizon          : forecast horizon
        Outputs
            hyperparameters, error metrics dataframe
        """
        alpha_err_dict = defaultdict(list)
        alpha = self.ALPHA_MIN
        while alpha <= self.ALPHA_MAX:
            rmse_mean, _ = self.get_error_metrics(series, train_size, horizon, alpha)
            alpha_err_dict['alpha'].append(alpha)
            alpha_err_dict['rmse'].append(rmse_mean)
            alpha = alpha + self.ALPHA_STEP
        alpha_err_df = pd.DataFrame(alpha_err_dict)
        rmse_min = alpha_err_df['rmse'].min()
        return alpha_err_df[alpha_err_df['rmse'] == rmse_min]['alpha'].values[0], alpha_err_df

    def predict(self):
        train_val_size = self.train_test_size
        i = train_val_size
        print("Predict for day %d, with horizon %d" % (i, self.horizon))
        # Predict
        print(i-train_val_size)
        print(i)
        preds_list = self.simple_single_exponential_smoothing(self.series[i - train_val_size:i], self.horizon)
        print("For horizon %d, predict for day %d, the RMSE is %f" % (
        self.horizon, i, self.get_rmse(self.series[i:i + self.horizon], preds_list)))

import sys
import numpy as np
from global_config import GlobalConfig

'''

Kalman Filter
Butterworth Filter
Moving Average
RMS
Exponential Smoothing
Independent Component Analysis

'''

class MovingAverage:

    def __init__(self, parser_object, time_series, window_size, weighted, weights):
        self.parser_object = parser_object
        self.time_series = time_series
        self.window_size = window_size
        self.weighted = weighted
        self.weights = weights
        self.moving_average_data_dict = dict({GlobalConfig.MOVING_AVG_STR: {}})

    def calculate_moving_average(self):

        # extract recording list of respective stock
        if self.parser_object.name == GlobalConfig.BITCOIN_STR:
            recording_list = self.parser_object.single_bitcoin_recording_list
        elif self.parser_object.name == GlobalConfig.GOOGLE_STR:
            recording_list = self.parser_object.single_google_recording_list
        else:
            print('Valid parameters for <stock> are "Bitcoin" and "Google".')
            sys.exit(0)

        # extract data
        time_series_list_original = []
        time_stamp_list_original = []
        for single_bitcoin_recording in recording_list:
            time_stamp_list_original.append(single_bitcoin_recording.time_stamp)
            if self.time_series == GlobalConfig.OPEN_STR:
                time_series_list_original.append(single_bitcoin_recording.open)
            elif self.time_series == GlobalConfig.LOW_STR:
                time_series_list_original.append(single_bitcoin_recording.low)
            elif self.time_series == GlobalConfig.HIGH_STR:
                time_series_list_original.append(single_bitcoin_recording.high)
            elif self.time_series == GlobalConfig.CLOSE_STR:
                time_series_list_original.append(single_bitcoin_recording.close)
            elif self.time_series == GlobalConfig.VOLUME_STR:
                time_series_list_original.append(single_bitcoin_recording.volume)
            else:
                print('Valid parameters for <time_series> are "open", "high", "low", "close" and "volume".')


        # perform normal moving average
        time_series_list_original = np.asarray(time_series_list_original)
        time_series_list_ma = []
        time_stamp_list_ma = []
        if not self.weighted:
            for idx in list(range(0, time_series_list_original.shape[0]-self.window_size+1)):
                average = np.nanmean(time_series_list_original[idx: idx+self.window_size])
                time_series_list_ma.append(average)
                time_stamp_list_ma.append(time_stamp_list_original[idx+(self.window_size // 2)])

        # perform weighted moving average
        else:
            if np.sum(np.asarray(self.weights)) != 1:
                print('Weights should add up to 1. ')
                sys.exit(0)
            elif len(self.weights) != self.window_size:
                print('Number of weights should correspond to the window size or set <weighted> to "False".')
                sys.exit(0)
            else:
                for idx in list(range(0, time_series_list_original.shape[0] - self.window_size + 1)):
                    average = np.dot(np.asarray(self.weights), time_series_list_original[idx: idx + self.window_size])
                    time_series_list_ma.append(average)
                    time_stamp_list_ma.append(time_stamp_list_original[idx + (self.window_size // 2)])

        # update moving_average_data_dict
        self.moving_average_data_dict.get(GlobalConfig.MOVING_AVG_STR).update({self.time_series: time_series_list_ma})
        self.moving_average_data_dict.get(GlobalConfig.MOVING_AVG_STR).update({GlobalConfig.TIMESTAMP_STR: time_stamp_list_ma})



class KalmanFilter():

    def __init__(self, parser_object, time_series, Q, R, prediction_time):
        self.parser_object = parser_object
        self.time_series = time_series
        self.Q = Q
        self.R = R
        self.prediction_time = prediction_time
        self.kalman_filter_dict = dict({GlobalConfig.KALMAN_FILTER: {}})


    def calculate_kalman_filter(self):

        # extract recording list of respective stock
        if self.parser_object.name == GlobalConfig.BITCOIN_STR:
            recording_list = self.parser_object.single_bitcoin_recording_list
        elif self.parser_object.name == GlobalConfig.GOOGLE_STR:
            recording_list = self.parser_object.single_google_recording_list
        else:
            print('Valid parameters for <stock> are "Bitcoin" and "Google".')
            sys.exit(0)

        # extract data
        time_series_list_original = []
        time_stamp_list_original = []
        for idx, single_bitcoin_recording in enumerate(recording_list):
            if idx % self.prediction_time == 0:
                time_stamp_list_original.append(single_bitcoin_recording.time_stamp)
                if self.time_series == GlobalConfig.OPEN_STR:
                    time_series_list_original.append(single_bitcoin_recording.open)
                elif self.time_series == GlobalConfig.LOW_STR:
                    time_series_list_original.append(single_bitcoin_recording.low)
                elif self.time_series == GlobalConfig.HIGH_STR:
                    time_series_list_original.append(single_bitcoin_recording.high)
                elif self.time_series == GlobalConfig.CLOSE_STR:
                    time_series_list_original.append(single_bitcoin_recording.close)
                elif self.time_series == GlobalConfig.VOLUME_STR:
                    time_series_list_original.append(single_bitcoin_recording.volume)
                else:
                    print('Valid parameters for <time_series> are "open", "high", "low", "close" and "volume".')
            else:
                continue
        # perform kalman filter
        num_iterations = len(time_series_list_original)
        z = time_series_list_original

        # initialize empty arrays
        x_hat = np.zeros(num_iterations)
        x_hat_minus = np.zeros(num_iterations)
        P = np.zeros(num_iterations)
        P_minus = np.zeros(num_iterations)
        K = np.zeros(num_iterations)

        # initial guesses
        x_hat[0] = 0
        P[0] = 1

        for k in range(1, num_iterations):

            # time update
            x_hat_minus[k] = x_hat[k-1]
            P_minus[k] = P_minus[k-1] + self.Q

            # measurement update
            K[k] = P_minus[k] / (P_minus[k] + self.R)
            x_hat[k] = x_hat_minus[k] + K[k]*(z[k]-x_hat_minus[k])
            P[k] = (1-K[k]) * P_minus[k]

        # update kalman_filter_dict
        self.kalman_filter_dict.get(GlobalConfig.KALMAN_FILTER).update({self.time_series: x_hat})
        self.kalman_filter_dict.get(GlobalConfig.KALMAN_FILTER).update({GlobalConfig.TIMESTAMP_STR: time_stamp_list_original})




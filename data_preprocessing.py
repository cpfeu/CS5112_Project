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

class PreprocessorObject:

    def __init__(self, parser_object):
        self.parser_object = parser_object
        self.preprocessed_data_dict = dict({GlobalConfig.MOVING_AVG_STR: {}})


    def moving_average(self, stock, time_series, window_size, weighted, weights):

        # extract recording list of respective stock
        if stock == GlobalConfig.BITCOIN_STR:
            recording_list = self.parser_object.single_bitcoin_recording_list
        elif stock == GlobalConfig.GOOGLE_STR:
            recording_list = self.parser_object.single_google_recording_list
        else:
            print('Valid parameters for <stock> are "Bitcoin" and "Google".')
            sys.exit(0)

        # extract data
        time_series_list_original = []
        time_stamp_list_original = []
        for single_bitcoin_recording in recording_list:
            time_stamp_list_original.append(single_bitcoin_recording.time_stamp)
            if time_series == GlobalConfig.OPEN_STR:
                time_series_list_original.append(single_bitcoin_recording.open)
            elif time_series == GlobalConfig.LOW_STR:
                time_series_list_original.append(single_bitcoin_recording.low)
            elif time_series == GlobalConfig.HIGH_STR:
                time_series_list_original.append(single_bitcoin_recording.high)
            elif time_series == GlobalConfig.CLOSE_STR:
                time_series_list_original.append(single_bitcoin_recording.close)
            elif time_series == GlobalConfig.VOLUME_STR:
                time_series_list_original.append(single_bitcoin_recording.volume)
            else:
                print('Valid parameters for <time_series> are "open", "high", "low", "close" and "volume".')


        # perform normal moving average
        time_series_list_original = np.asarray(time_series_list_original)
        time_series_list_ma = []
        time_stamp_list_ma = []
        if not weighted:
            for idx in list(range(0, time_series_list_original.shape[0]-window_size+1)):
                average = np.nanmean(time_series_list_original[idx: idx+window_size])
                time_series_list_ma.append(average)
                time_stamp_list_ma.append(time_stamp_list_original[idx+(window_size // 2)])

        # perform weighted moving average
        else:
            if np.sum(np.asarray(weights)) != 1:
                print('Weights should add up to 1. ')
                sys.exit(0)
            else:
                for idx in list(range(0, time_series_list_original.shape[0] - window_size + 1)):
                    average = np.dot(np.asarray(weights), time_series_list_original[idx: idx + window_size])
                    time_series_list_ma.append(average)
                    time_stamp_list_ma.append(time_stamp_list_original[idx + (window_size // 2)])

        # update preprocessed_data_dict
        self.preprocessed_data_dict.get(GlobalConfig.MOVING_AVG_STR).update({time_series: time_series_list_ma})
        self.preprocessed_data_dict.get(GlobalConfig.MOVING_AVG_STR).update({GlobalConfig.TIMESTAMP_STR: time_stamp_list_ma})






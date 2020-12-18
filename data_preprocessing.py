import sys
import numpy as np
import matplotlib.pyplot as plt
from global_config import GlobalConfig
from statsmodels.tsa.seasonal import seasonal_decompose

'''
Kalman Filter
Butterworth Filter
Moving Average
RMS
Exponential Smoothing
Independent Component Analysis
'''

class MovingAverage:

    def __init__(self, parser_object, time_series, window_size, weighted, weights, modify_time_series=False):

        '''
        This is the constructor of a <MovingAverage> object that takes in the following parameters:
        :param parser_object: <BitcoinParser> or <GoogleParser>: object holding the parsed data
        :param time_series: <str> - either 'open', 'close, 'low', 'high' or 'volume'
        :param window_size: <int> - integer specifying m (the window siz) of the moving average function.
                                    Ideally, this parameter should be an odd number so that the moving average function
                                    is symmetric.
        :param weighted: <boolean> - if True:, custom weights are required to perform a customized moving average
        :param modify_time_series: <boolean> - if True, only extract every 360th value to compare with Kalman Filer
        :param weights: <list> - a list of floats that sum up to 1 and has the length of window_size
        moving_average_data_dict: <dict> - dictionary in which the calculated moving average time series is stored
        '''

        self.parser_object = parser_object
        self.time_series = time_series
        self.window_size = window_size
        self.weighted = weighted
        self.weights = weights
        self.modify_time_series = modify_time_series
        self.moving_average_data_dict = dict({GlobalConfig.MOVING_AVG_STR: {}})


    def calculate_moving_average(self):

        '''
        The moving average function extracts the relevant original time series and the corresponding list
        of timestamps. While the function performs the moving average, the newly created list of prices and the
        newly created list of corresponding timestamps are stored in two new arrays. These arrays are stored
        in the dictionary of the <MovingAverage> object.
        :return:
        '''

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

        # modify original time series
        if self.modify_time_series:
            time_series_list_original_shortened = []
            time_stamp_list_original_shortened = []
            idx = 0
            for timestamp, price in zip(time_stamp_list_original, time_series_list_original):
                if idx % 360 == 0:
                    time_stamp_list_original_shortened.append(timestamp)
                    time_series_list_original_shortened.append(price)
                idx += 1
            time_stamp_list_original = time_stamp_list_original_shortened
            time_series_list_original = time_series_list_original_shortened

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
                for idx in list(range(0, time_series_list_original.shape[0]-self.window_size+1)):
                    average = np.dot(np.asarray(self.weights), time_series_list_original[idx: idx + self.window_size])
                    time_series_list_ma.append(average)
                    time_stamp_list_ma.append(time_stamp_list_original[idx + (self.window_size // 2)])

        # update moving_average_data_dict
        self.moving_average_data_dict.get(GlobalConfig.MOVING_AVG_STR).update({self.time_series: time_series_list_ma})
        self.moving_average_data_dict.get(GlobalConfig.MOVING_AVG_STR).update({GlobalConfig.TIMESTAMP_STR: time_stamp_list_ma})





class KalmanFilter:

    def __init__(self, parser_object, time_series, Q, R, prediction_time):

        '''
        This is the constructor of a <KalmanFilter> object that takes in the following parameters:
        :param parser_object: <BitcoinParser> or <GoogleParser>: object holding the parsed data
        :param time_series: <str> - either 'open', 'close, 'low', 'high' or 'volume'
        :param Q: <float> - process variance
        :param R: <flot> - estimate of measurement variance
        :param prediction_time: <int> - only every 'prediction_time' recording is used to create a time series
                                        that is being used for the Kalman Filter. Otherwise, the preprocessed Kalman
                                        Filter time series will be almost exactly the same and could not serve as a
                                        trend extractor.
        kalman_filter_dict: <dict> - dictionary in which the calculated kalman filter time series is stored
        '''

        self.parser_object = parser_object
        self.time_series = time_series
        self.Q = Q
        self.R = R
        self.prediction_time = prediction_time
        self.kalman_filter_dict = dict({GlobalConfig.KALMAN_FILTER: {}})


    def calculate_kalman_filter(self):

        '''
        The kalman filter function extracts the relevant original time series and the corresponding list
        of timestamps. It only extracts every 'self.prediction_time' records as otherwise the Kalman Filter has almost
        no effect since the individual records are too close to each other.
        While the function performs the kalman filter, the newly created list of prices and the
        newly created list of corresponding timestamps are stored in two new arrays. These arrays are stored
        in the dictionary of the <KalmanFilter> object.
        :return:
        '''

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
        for idx, single_recording in enumerate(recording_list):
            if idx % self.prediction_time == 0:
                time_stamp_list_original.append(single_recording.time_stamp)
                if self.time_series == GlobalConfig.OPEN_STR:
                    time_series_list_original.append(single_recording.open)
                elif self.time_series == GlobalConfig.LOW_STR:
                    time_series_list_original.append(single_recording.low)
                elif self.time_series == GlobalConfig.HIGH_STR:
                    time_series_list_original.append(single_recording.high)
                elif self.time_series == GlobalConfig.CLOSE_STR:
                    time_series_list_original.append(single_recording.close)
                elif self.time_series == GlobalConfig.VOLUME_STR:
                    time_series_list_original.append(single_recording.volume)
                else:
                    print('Valid parameters for <time_series> are "open", "high", "low", "close" and "volume".')
            else:
                continue

        # perform kalman filter
        '''
        https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
        '''
        num_iterations = len(time_series_list_original)
        z = time_series_list_original

        # initialize empty arrays
        x_hat = np.zeros(num_iterations)        # a posteri estimate of x
        x_hat_minus = np.zeros(num_iterations)  # a priori estimate of x
        P = np.zeros(num_iterations)            # a posteri error estimate
        P_minus = np.zeros(num_iterations)      # a priori error estimate
        K = np.zeros(num_iterations)            # gain or blending factor

        # initial guesses
        x_hat[0] = 0
        P[0] = 1

        for k in range(1, num_iterations):

            'https: // scipy - cookbook.readthedocs.io / items / KalmanFiltering.html'

            # time update
            x_hat_minus[k] = x_hat[k-1]
            P_minus[k] = P[k-1] + self.Q

            # measurement update
            K[k] = P_minus[k] / (P_minus[k] + self.R)
            x_hat[k] = x_hat_minus[k] + K[k]*(z[k]-x_hat_minus[k])
            P[k] = (1-K[k]) * P_minus[k]


        # update kalman_filter_dict
        self.kalman_filter_dict.get(GlobalConfig.KALMAN_FILTER).update({self.time_series: x_hat})
        self.kalman_filter_dict.get(GlobalConfig.KALMAN_FILTER).update({GlobalConfig.TIMESTAMP_STR: time_stamp_list_original})




class ExponentialSmoothing():

    def __init__(self, parser_object, time_series):
        self.parser_object = parser_object
        self.horizon = 1
        time_stamp_list = []
        close_list = []
        for single_google_recording in parser_object.single_google_recording_list:
            time_stamp_list.append(single_google_recording.time_stamp)
            close_list.append(single_google_recording.close)
        self.series = close_list
        self.time_series = time_series
        self.time_stamp_list = time_stamp_list
        self.exponential_smoothing_dict = dict({GlobalConfig.EXPONENTIAL_SMOOTHING: {}})

    def single_exponential_smoothing(self, series, horizon, alpha=0.5):
        result = [0, series[0]]
        for i in range(1, len(series) + horizon - 1):
            if i >= len(series):
                result.append((series[-1] * alpha) + ((1-alpha) * result[i]))
            else:
                result.append((series[i] * alpha) + ((1-alpha) * result[i]))
        return result[len(series):len(series)+horizon]

    def double_exponential_smoothing(self, series, horizon, alpha=0.5, beta=0.5):
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

    def calculate_single_exponential_smoothing(self):
        if self.parser_object.name == GlobalConfig.BITCOIN_STR:
            recording_list = self.parser_object.single_bitcoin_recording_list
        elif self.parser_object.name == GlobalConfig.GOOGLE_STR:
            recording_list = self.parser_object.single_google_recording_list
        else:
            print('Valid parameters for <stock> are "Bitcoin" and "Google".')
            sys.exit(0)
        preds = []
        for i in range(1, len(self.series)+1):
            preds.append(self.single_exponential_smoothing(self.series[:i], 1)[0])

        self.exponential_smoothing_dict.get(GlobalConfig.EXPONENTIAL_SMOOTHING).update({self.time_series: preds})
        self.exponential_smoothing_dict.get(GlobalConfig.EXPONENTIAL_SMOOTHING).update({GlobalConfig.TIMESTAMP_STR: self.time_stamp_list})

    def calculate_double_exponential_smoothing(self):
        if self.parser_object.name == GlobalConfig.BITCOIN_STR:
            recording_list = self.parser_object.single_bitcoin_recording_list
        elif self.parser_object.name == GlobalConfig.GOOGLE_STR:
            recording_list = self.parser_object.single_google_recording_list
        else:
            print('Valid parameters for <stock> are "Bitcoin" and "Google".')
            sys.exit(0)

        preds = []
        for i in range(2, len(self.series)+2):
            preds.append(self.double_exponential_smoothing(self.series[:i], 1)[0])

        self.exponential_smoothing_dict.get(GlobalConfig.EXPONENTIAL_SMOOTHING).update({self.time_series: preds})
        self.exponential_smoothing_dict.get(GlobalConfig.EXPONENTIAL_SMOOTHING).update({GlobalConfig.TIMESTAMP_STR: self.time_stamp_list})




class DecomposeTimeSeries():

    def __init__(self, parser_object, time_series, decompose_model, period):

        '''
        This is the constructor of a <DecomposeTimeSeries> object that takes in the following parameters:
        :param parser_object: <BitcoinParser> or <GoogleParser>: object holding the parsed data
        :param time_series: <str> - either 'open', 'close, 'low', 'high' or 'volume'
        :param decompose_model: <str> - either additive or multiplicative
        :param period: <int> - the frequency of the time series. This value will influence the strength of
                               seasonal component and trend component
        '''

        self.parser_object = parser_object
        self.time_series = time_series
        self.decompose_model = decompose_model
        self.period = period
        self.decomposed_time_series = None


    def decompose_time_series(self, decompose_with_kalman_filter=False,
                              decompose_with_moving_average=False, show_decomposed_ts=True):

        '''
        The moving decompose function extracts the relevant original time series and the corresponding list
        of timestamps. It then decomposes the original time series or a preprocessed time series
        :param decompose_with_kalman_filter: <boolean> if True, a <KalmanFilter> object is created and the
                                             extracted time series is preprocessed with a Kalman Filter
        :param decompose_with_moving_average: <boolean> if True, a <MovingAverage> object is created and the
                                              extracted time series is preprocessed with a moving average
        :param show_decomposed_ts: <boolean> if True, show a plot of the trend component, the seasonal component and
                                             the remainder component
        :return:
        '''

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
        for single_recording in recording_list:
            time_stamp_list_original.append(single_recording.time_stamp)
            if self.time_series == GlobalConfig.OPEN_STR:
                time_series_list_original.append(single_recording.open)
            elif self.time_series == GlobalConfig.LOW_STR:
                time_series_list_original.append(single_recording.low)
            elif self.time_series == GlobalConfig.HIGH_STR:
                time_series_list_original.append(single_recording.high)
            elif self.time_series == GlobalConfig.CLOSE_STR:
                time_series_list_original.append(single_recording.close)
            elif self.time_series == GlobalConfig.VOLUME_STR:
                time_series_list_original.append(single_recording.volume)
            else:
                print('Valid parameters for <time_series> are "open", "high", "low", "close" and "volume".')

        # calculate decomposition
        if decompose_with_kalman_filter:
            kal_obj = KalmanFilter(parser_object=self.parser_object,
                                   time_series=self.time_series,
                                   Q=1e-5, R=0.1 ** 2, prediction_time=300)
            kal_obj.calculate_kalman_filter()
            self.kalman_object = kal_obj
            self.decomposed_time_series = seasonal_decompose(x=self.kalman_object.kalman_filter_dict.
                                                             get(GlobalConfig.KALMAN_FILTER).get(self.time_series)[::-1],
                                                             model=self.decompose_model, period=self.period)

        elif decompose_with_moving_average:
            ma_obj = MovingAverage(parser_object=self.parser_object,
                                   time_series=self.time_series,
                                   window_size=7501,
                                   weighted=False,
                                   weights=[0.1, 0.2, 0.3, 0.4],
                                   modify_time_series=False)
            ma_obj.calculate_moving_average()
            self.ma_obj = ma_obj
            self.decomposed_time_series = seasonal_decompose(x=self.ma_obj.moving_average_data_dict.
                                                             get(GlobalConfig.MOVING_AVG_STR).get(self.time_series)[::-1],
                                                             model=self.decompose_model, period=self.period)

        else:
            self.decomposed_time_series = seasonal_decompose(x=time_series_list_original[::-1],
                                                             model=self.decompose_model, period=self.period)

        # show decomposed time series
        if show_decomposed_ts:
            self.decomposed_time_series.plot()
            plt.show()

        # measure trend strength and seasonal strength of time series
        try:
            trend_strength_temp = 1 - ((np.nanstd(self.decomposed_time_series.resid))**2 /
                                       (np.nanstd(self.decomposed_time_series.resid + self.decomposed_time_series.trend))**2)
        except RuntimeWarning:
            trend_strength_temp = 1
        except ZeroDivisionError:
            trend_strength_temp = 1
        trend_strength = np.amax([0, trend_strength_temp])
        print('Trend strength: ', trend_strength)

        try:
            seasonal_strength_temp = 1 - ((np.nanstd(self.decomposed_time_series.resid))**2 /
                                          (np.nanstd(self.decomposed_time_series.resid + self.decomposed_time_series.seasonal))**2)
        except RuntimeWarning:
            seasonal_strength_temp = 1
        except ZeroDivisionError:
            seasonal_strength_temp = 1
        seasonal_strength = np.amax([0, seasonal_strength_temp])
        print('Seasonal strength: ', seasonal_strength)

        print()
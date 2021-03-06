import os
import numpy as np
import plotly.offline as po
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from datetime import datetime
from global_config import GlobalConfig
from statsmodels.graphics.tsaplots import plot_acf



class BitcoinVisualizer:

    def __init__(self, parser_object, preprocessor_object=None, forecaster_object=None):

        '''
        This is the constructor for a <BitcoinVisualizer> object that holds the following parameter:
        :param parser_object: <BitcoinParser> or <GoogleParser> - object that holds original time series data
        :param preprocessor_object: <KalmanFilter>, <MovingAverage>, ... - object that holds preprocessed time series data
        :param forecaster_object: <SupportVectorRegression>, ... - object that holds predicted time series data
        '''

        self.parser_object = parser_object
        self.preprocessor_object = preprocessor_object
        self.forecaster_object = forecaster_object

    def plot_all_in_one_chart(self):

        '''
        This function plots the raw time series sored in the parser_object.
        It plots a number of 5 time series into one plot: open, close, low, high and volume
        :return:
        '''

        time_stamp_list = []
        open_list = []
        close_list = []
        high_list = []
        low_list = []
        volume_list = []
        for single_bitcoin_recording in self.parser_object.single_bitcoin_recording_list:
            time_stamp_list.append(single_bitcoin_recording.time_stamp)
            open_list.append(single_bitcoin_recording.open)
            high_list.append(single_bitcoin_recording.high)
            close_list.append(single_bitcoin_recording.close)
            volume_list.append(single_bitcoin_recording.volume)
            low_list.append(single_bitcoin_recording.low)

        # create traces
        open_trace = go.Scattergl(x=time_stamp_list, y=open_list, mode='lines', name=GlobalConfig.OPEN_STR)
        close_trace = go.Scattergl(x=time_stamp_list, y=close_list, mode='lines', name=GlobalConfig.CLOSE_STR)
        high_trace = go.Scattergl(x=time_stamp_list, y=high_list, mode='lines', name=GlobalConfig.HIGH_STR)
        low_trace = go.Scattergl(x=time_stamp_list, y=low_list, mode='lines', name=GlobalConfig.LOW_STR)
        volume_trace = go.Scattergl(x=time_stamp_list, y=volume_list, mode='lines', name=GlobalConfig.VOLUME_STR)

        # create and plot figure
        figure = dict(data=[open_trace, close_trace, high_trace, low_trace, volume_trace])
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.BITCOIN_STR, "All_in_one_plot.html"), auto_open=False)
        print(datetime.now(), ': all_in_one_plot created.')

    def plot_moving_average(self):

        '''
        This function plots the calculated moving average against the original time series.
        :return:
        '''

        # extract data
        time_series_list_original = []
        time_stamp_list_original = []
        for single_bitcoin_recording in self.parser_object.single_bitcoin_recording_list:
            time_stamp_list_original.append(single_bitcoin_recording.time_stamp)
            if self.preprocessor_object.time_series == GlobalConfig.OPEN_STR:
                time_series_list_original.append(single_bitcoin_recording.open)
            elif self.preprocessor_object.time_series == GlobalConfig.LOW_STR:
                time_series_list_original.append(single_bitcoin_recording.low)
            elif self.preprocessor_object.time_series == GlobalConfig.HIGH_STR:
                time_series_list_original.append(single_bitcoin_recording.high)
            elif self.preprocessor_object.time_series == GlobalConfig.CLOSE_STR:
                time_series_list_original.append(single_bitcoin_recording.close)
            elif self.preprocessor_object.time_series == GlobalConfig.VOLUME_STR:
                time_series_list_original.append(single_bitcoin_recording.volume)
            else:
                print('Valid parameters for <time_series> are "open", "high", "low", "close" and "volume".')

        time_series_list_original_shortened = []
        time_stamp_list_original_shortened = []
        if self.preprocessor_object.modify_time_series:
            idx = 0
            for timestamp, price in zip(time_stamp_list_original, time_series_list_original):
                if idx % 360 == 0:
                    time_stamp_list_original_shortened.append(timestamp)
                    time_series_list_original_shortened.append(price)
                idx += 1
        time_stamp_list_original = time_stamp_list_original_shortened
        time_series_list_original = time_series_list_original_shortened

        time_series_list_ma = self.preprocessor_object.moving_average_data_dict.\
            get(GlobalConfig.MOVING_AVG_STR).get(self.preprocessor_object.time_series)
        time_stamp_list_ma = self.preprocessor_object.moving_average_data_dict.\
            get(GlobalConfig.MOVING_AVG_STR).get(GlobalConfig.TIMESTAMP_STR)

        # create traces
        open_trace_original = go.Scattergl(x=time_stamp_list_original, y=time_series_list_original, mode='lines',
                                           name=self.preprocessor_object.time_series,
                                           opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')
        open_trace_ma = go.Scattergl(x=time_stamp_list_ma, y=time_series_list_ma, mode='lines',
                                     name='moving average - window size='+str(self.preprocessor_object.window_size),
                                     opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')

        # design layout
        layout = dict(title='Bitcoin Moving Average',
                      xaxis=dict(title='Time',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      yaxis=dict(title='Price [in $]',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      hovermode='closest')

        # create and plot figure
        figure = dict(data=[open_trace_original, open_trace_ma], layout=layout)
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.BITCOIN_STR, "moving_average_plot.html"), auto_open=False)
        print(datetime.now(), ': moving_average_plot created.')


    def plot_kalman_filter(self):

        '''
        This function plots the calculated Kalman Filter times series against the original time series.
        :return:
        '''

        # extract data
        time_series_list_original = []
        time_stamp_list_original = []
        for idx, single_bitcoin_recording in enumerate(self.parser_object.single_bitcoin_recording_list):
            if idx % self.preprocessor_object.prediction_time == 0:
                time_stamp_list_original.append(single_bitcoin_recording.time_stamp)
                if self.preprocessor_object.time_series == GlobalConfig.OPEN_STR:
                    time_series_list_original.append(single_bitcoin_recording.open)
                elif self.preprocessor_object.time_series == GlobalConfig.LOW_STR:
                    time_series_list_original.append(single_bitcoin_recording.low)
                elif self.preprocessor_object.time_series == GlobalConfig.HIGH_STR:
                    time_series_list_original.append(single_bitcoin_recording.high)
                elif self.preprocessor_object.time_series == GlobalConfig.CLOSE_STR:
                    time_series_list_original.append(single_bitcoin_recording.close)
                elif self.preprocessor_object.time_series == GlobalConfig.VOLUME_STR:
                    time_series_list_original.append(single_bitcoin_recording.volume)
                else:
                    print('Valid parameters for <time_series> are "open", "high", "low", "close" and "volume".')
            else:
                continue

        time_series_list_kalman = self.preprocessor_object.kalman_filter_dict. \
            get(GlobalConfig.KALMAN_FILTER).get(self.preprocessor_object.time_series)
        time_stamp_list_kalman = self.preprocessor_object.kalman_filter_dict. \
            get(GlobalConfig.KALMAN_FILTER).get(GlobalConfig.TIMESTAMP_STR)

        # create traces
        open_trace_original = go.Scattergl(x=time_stamp_list_original, y=time_series_list_original, mode='lines',
                                           name=self.preprocessor_object.time_series,
                                           opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')
        open_trace_kalman = go.Scattergl(x=time_stamp_list_kalman, y=time_series_list_kalman, mode='lines',
                                     name='kalman filter',
                                     opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')

        # design layout
        layout = dict(title='Bitcoin Kalman Filter',
                      xaxis=dict(title='Time',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      yaxis=dict(title='Price [in $]',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      hovermode='closest')

        # create and plot figure
        figure = dict(data=[open_trace_original, open_trace_kalman], layout=layout)
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.BITCOIN_STR, "kalman_filter_plot.html"),
                auto_open=False)
        print(datetime.now(), ': kalman_filter_plot created.')


    def plot_autocorrelation(self, lags=500):

        '''
        Time function extracts time series data
        :param lags: <int> - autocorrelation of a time series is calculated for a time lag of 1 up to this value
        :return:
        '''

        time_series = []
        for idx, single_bitcoin_recording in enumerate(self.parser_object.single_bitcoin_recording_list):
            if idx % 360 == 0:
                time_series.append(single_bitcoin_recording.close)

        plot_acf(x=time_series, lags=lags, alpha=None, use_vlines=True,
                 title='Bitcoin Autocorrelation: Lag: 1=6h', zero=True)
        plt.show()

        print(datetime.now(), ': autocorrelation_plot created.')





    def plot_svr_performace(self):

        dates = np.concatenate((self.forecaster_object.train_dates, self.forecaster_object.test_dates), axis=0)
        actual_prices = np.concatenate((self.forecaster_object.y_train, self.forecaster_object.y_test), axis=0)
        predictions = self.forecaster_object.predictions
        # create traces
        actual_price_trace = go.Scattergl(x=dates, y=actual_prices, mode='lines',
                                           name='Actual Prices',
                                           opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')
        predicted_price_trace = go.Scattergl(x=dates, y=predictions, mode='lines',
                                         name='Predicted prices',
                                         opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')

        # design layout
        layout = dict(title='SVR Regression Predictions',
                      xaxis=dict(title='Date',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      yaxis=dict(title='Price [in $]',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      hovermode='closest')

        # create and plot figure
        figure = dict(data=[actual_price_trace, predicted_price_trace], layout=layout)
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.BITCOIN_STR, "SVR_predictions_plot.html"),
                auto_open=False)
        print(datetime.now(), ': SVR_predictions_plot created.')







class GoogleVisualizer:

    def __init__(self, parser_object, preprocessor_object=None, forecaster_object=None):

        '''
        This is the constructor for a <GoogleVisualizer> object that holds the following parameter:
        :param parser_object: <BitcoinParser> or <GoogleParser> - object that holds original time series data
        :param preprocessor_object: <KalmanFilter>, <MovingAverage>, ... - object that holds preprocessed time series data
        :param forecaster_object: <SupportVectorRegression>, ... - object that holds predicted time series data
        '''

        self.parser_object = parser_object
        self.preprocessor_object = preprocessor_object
        self.forecaster_object = forecaster_object

    def plot_all_in_one_chart(self):

        '''
        This function plots the raw time series sored in the parser_object.
        It plots a number of 5 time series into one plot: open, close, low, high and volume
        :return:
        '''

        time_stamp_list = []
        open_list = []
        close_list = []
        high_list = []
        low_list = []
        volume_list = []
        for single_google_recording in self.parser_object.single_google_recording_list:
            time_stamp_list.append(single_google_recording.time_stamp)
            open_list.append(single_google_recording.open)
            high_list.append(single_google_recording.high)
            close_list.append(single_google_recording.close)
            volume_list.append(single_google_recording.volume)
            low_list.append(single_google_recording.low)

        # create traces
        open_trace = go.Scattergl(x=time_stamp_list, y=open_list, mode='lines', name=GlobalConfig.OPEN_STR)
        close_trace = go.Scattergl(x=time_stamp_list, y=close_list, mode='lines', name=GlobalConfig.CLOSE_STR)
        high_trace = go.Scattergl(x=time_stamp_list, y=high_list, mode='lines', name=GlobalConfig.HIGH_STR)
        low_trace = go.Scattergl(x=time_stamp_list, y=low_list, mode='lines', name=GlobalConfig.LOW_STR)
        volume_trace = go.Scattergl(x=time_stamp_list, y=volume_list, mode='lines', name=GlobalConfig.VOLUME_STR)

        # create and plot figure
        figure = dict(data=[open_trace, close_trace, high_trace, low_trace, volume_trace])
        if self.parser_object.data_path == GlobalConfig.GOOGLE_DATA_PATH:
            filename = 'All_in_one_plot.html'
        else:
            filename = 'All_in_one_plot_extended.html'
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.GOOGLE_STR, filename), auto_open=False)
        print(datetime.now(), ': all_in_one_plot created.')



    def plot_moving_average(self):

        '''
        This function plots the calculated moving average against the original time series.
        :return:
        '''

        # extract data
        time_series_list_original = []
        time_stamp_list_original = []
        for single_google_recording in self.parser_object.single_google_recording_list:
            time_stamp_list_original.append(single_google_recording.time_stamp)
            if self.preprocessor_object.time_series == GlobalConfig.OPEN_STR:
                time_series_list_original.append(single_google_recording.open)
            elif self.preprocessor_object.time_series == GlobalConfig.LOW_STR:
                time_series_list_original.append(single_google_recording.low)
            elif self.preprocessor_object.time_series == GlobalConfig.HIGH_STR:
                time_series_list_original.append(single_google_recording.high)
            elif self.preprocessor_object.time_series == GlobalConfig.CLOSE_STR:
                time_series_list_original.append(single_google_recording.close)
            elif self.preprocessor_object.time_series == GlobalConfig.VOLUME_STR:
                time_series_list_original.append(single_google_recording.volume)
            else:
                print('Valid parameters for <time_series> are "open", "high", "low", "close" and "volume".')

        time_series_list_original_shortened = []
        time_stamp_list_original_shortened = []
        if self.preprocessor_object.modify_time_series:
            idx = 0
            for timestamp, price in zip(time_stamp_list_original, time_series_list_original):
                if idx % 360 == 0:
                    time_stamp_list_original_shortened.append(timestamp)
                    time_series_list_original_shortened.append(price)
                idx += 1
        time_stamp_list_original = time_stamp_list_original_shortened
        time_series_list_original = time_series_list_original_shortened


        time_series_list_ma = self.preprocessor_object.moving_average_data_dict. \
            get(GlobalConfig.MOVING_AVG_STR).get(self.preprocessor_object.time_series)
        time_stamp_list_ma = self.preprocessor_object.moving_average_data_dict. \
            get(GlobalConfig.MOVING_AVG_STR).get(GlobalConfig.TIMESTAMP_STR)

        # create traces
        open_trace_original = go.Scattergl(x=time_stamp_list_original, y=time_series_list_original,
                                           mode='lines', name=self.preprocessor_object.time_series,
                                           opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')
        open_trace_ma = go.Scattergl(x=time_stamp_list_ma, y=time_series_list_ma, mode='lines',
                                     name='moving average - window size='+str(self.preprocessor_object.window_size),
                                     opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')

        # design layout
        layout = dict(title='Google Moving Average',
                      xaxis=dict(title='Time',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      yaxis=dict(title='Price [in $]',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      hovermode='closest')

        # create and plot figure
        figure = dict(data=[open_trace_original, open_trace_ma], layout=layout)
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.GOOGLE_STR, "moving_average_plot.html"),
                auto_open=False)
        print(datetime.now(), ': moving_average_plot created.')


    def plot_kalman_filter(self):

        '''
        This function plots the calculated Kalman Filter times series against the original time series.
        :return:
        '''

        # extract data
        time_series_list_original = []
        time_stamp_list_original = []
        for idx, single_google_recording in enumerate(self.parser_object.single_google_recording_list):
            if idx % self.preprocessor_object.prediction_time == 0:
                time_stamp_list_original.append(single_google_recording.time_stamp)
                if self.preprocessor_object.time_series == GlobalConfig.OPEN_STR:
                    time_series_list_original.append(single_google_recording.open)
                elif self.preprocessor_object.time_series == GlobalConfig.LOW_STR:
                    time_series_list_original.append(single_google_recording.low)
                elif self.preprocessor_object.time_series == GlobalConfig.HIGH_STR:
                    time_series_list_original.append(single_google_recording.high)
                elif self.preprocessor_object.time_series == GlobalConfig.CLOSE_STR:
                    time_series_list_original.append(single_google_recording.close)
                elif self.preprocessor_object.time_series == GlobalConfig.VOLUME_STR:
                    time_series_list_original.append(single_google_recording.volume)
                else:
                    print('Valid parameters for <time_series> are "open", "high", "low", "close" and "volume".')
            else:
                continue

        time_series_list_kalman = self.preprocessor_object.kalman_filter_dict. \
            get(GlobalConfig.KALMAN_FILTER).get(self.preprocessor_object.time_series)
        time_stamp_list_kalman = self.preprocessor_object.kalman_filter_dict. \
            get(GlobalConfig.KALMAN_FILTER).get(GlobalConfig.TIMESTAMP_STR)

        # create traces
        open_trace_original = go.Scattergl(x=time_stamp_list_original, y=time_series_list_original, mode='lines',
                                           name=self.preprocessor_object.time_series,
                                           opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')
        open_trace_kalman = go.Scattergl(x=time_stamp_list_kalman, y=time_series_list_kalman, mode='lines',
                                     name='kalman filter',
                                     opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')

        # design layout
        layout = dict(title='Google Kalman Filter',
                      xaxis=dict(title='Time',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      yaxis=dict(title='Price [in $]',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      hovermode='closest')

        # create and plot figure
        figure = dict(data=[open_trace_original, open_trace_kalman], layout=layout)
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.GOOGLE_STR, "kalman_filter_plot.html"),
                auto_open=False)
        print(datetime.now(), ': kalman_filter_plot created.')



    def plot_autocorrelation(self, lags):

        '''
        Time function extracts time series data
        :param lags: <int> - autocorrelation of a time series is calculated for a time lag of 1 up to this value
        :return:
        '''

        high_list = []
        for idx, single_google_recording in enumerate(self.parser_object.single_google_recording_list):
            if idx % 360 == 0:
                high_list.append(single_google_recording.close)


        plot_acf(x=high_list, lags=lags, alpha=None, use_vlines=True, title='Google Autocorrelation: Lag: 1=6h', zero=True)
        plt.show()

        print(datetime.now(), ': autocorrelation_plot created.')



    def plot_svr_performace(self):

        dates = np.concatenate((self.forecaster_object.train_dates, self.forecaster_object.test_dates), axis=0)
        actual_prices = np.concatenate((self.forecaster_object.y_train, self.forecaster_object.y_test), axis=0)
        predictions = self.forecaster_object.predictions
        # create traces
        actual_price_trace = go.Scattergl(x=dates, y=actual_prices, mode='lines',
                                           name='Actual Prices',
                                           opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')
        predicted_price_trace = go.Scattergl(x=dates, y=predictions, mode='lines',
                                         name='Predicted prices',
                                         opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')

        # design layout
        layout = dict(title='SVR Regression Predictions',
                      xaxis=dict(title='Date',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      yaxis=dict(title='Price [in $]',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      hovermode='closest')

        # create and plot figure
        figure = dict(data=[actual_price_trace, predicted_price_trace], layout=layout)
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.GOOGLE_STR, "SVR_predictions_plot.html"),
                auto_open=False)
        print(datetime.now(), ': SVR_predictions_plot created.')

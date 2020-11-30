from datetime import datetime
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os
import plotly.offline as po
import plotly.graph_objs as go
from global_config import GlobalConfig
from sklearn.svm import SVR
from data_error_analysis import ErrorAnalyzer


class SimpleExponentialSmoothingForecaster:

    def __init__(self, parser_object=None, time_series=None, time_stamps=None):
        if parser_object:
            self.parser_object = parser_object
            self.horizon = 1
            time_stamp_list = []
            close_list = []
            for single_google_recording in parser_object.single_google_recording_list:
                time_stamp_list.append(single_google_recording.time_stamp)
                close_list.append(single_google_recording.close)

            print(datetime.now(), ': Exponential Smoothing Model Received Data')
            self.series = close_list
            self.time_stamp_list = time_stamp_list
        elif time_series and time_stamps:
            self.series = time_series
            self.time_stamp_list = time_stamps
            print(datetime.now(), ': Expontential Smoothing Received Data')
        else:
            print(datetime.now(), ': Exponential Smoothing Did Not Receive Data')

    def single_exponential_smoothing(self, series, horizon, alpha=0.5):
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

    def double_exponential_smoothing(self, series, horizon, alpha=0.5, beta=0.5):
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

    def triple_exponential_smoothing(self, series, t, horizon, alpha=0.3, beta=0.3, gamma=0.3):
        result = [0, series[0]]
        smooth = series[0]
        trend = self.trend(series, t)
        seasonals = self.create_seasonals(series, t)
        seasonals.append(seasonals[0])
        for n in range(1, len(series)+horizon-1):
            if n >= len(series):
                m = n - len(series) + 2
                result.append(smooth + m*trend + seasonals[n+1])
            else:
                val = series[n]
                last_smooth, smooth = smooth, alpha*(val-seasonals[n]) + (1-alpha)*(smooth+trend)
                trend = beta * (smooth-last_smooth) + (1-beta)*trend
                seasonals.append(gamma*(val-smooth) + (1-gamma)*seasonals[n])
                result.append(smooth + trend + seasonals[n+1])

        return result[len(series):len(series)+horizon]

    def create_seasonals(self, series, t):
        avgs, snls = [], []
        n = int(len(series)/t)
        for j in range(n):
            avgs.append(sum(series[t*j:t*j+t])/float(t))
        for i in range(t):
            sum_of_vals_over_avg = 0.0
            for j in range(n):
                sum_of_vals_over_avg += series[t*j+i]-avgs[j]
            snls.append(sum_of_vals_over_avg/n)
        return snls

    def trend(self, series, t):
        sum = 0.0
        for i in range(t):
            sum = sum + (float(series[i+t] - series[i]) / t)
        return sum / t

    def predict(self, seasons=4, alpha=0.5, beta=0.5, gamma=0.5, type="single", plot=True):
        preds = []
        steps = len(self.series)+2
        stamps = [int(steps/10)*i for i in range(1, 10)]
        for i in range(1, steps):
            if type == "single":
                preds.append(self.single_exponential_smoothing(self.series[:i], 1, alpha))
            elif type == "double" and i > 1:
                preds.append(self.double_exponential_smoothing(self.series[:i], 1, alpha, beta))
            elif type == "triple":
                preds.append(self.triple_exponential_smoothing(self.series[:i], seasons, 1, alpha, beta, gamma))
            if i in stamps:
                print(str(datetime.now()) + ': Exponential Smoothing Prediction {}0% Complete'.format(stamps.index(i)+1))
        print(datetime.now(), 'Successful Exponential Smoothing Prediction for {}'.format(type))
        predictions = []
        for pred in preds:
            predictions.append(pred[0])
        if not plot:
            return predictions

        open_trace_original = go.Scattergl(x=self.time_stamp_list, y=self.series, mode='lines',
                                           name='original',
                                           opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')
        open_trace_ma = go.Scattergl(x=self.time_stamp_list, y=predictions, mode='lines',
                                     name='exponential smoothing - type: '+type,
                                     opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')
        # design layout
        layout = dict(title='Exponential Smoothing',
                      xaxis=dict(title='Time',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      yaxis=dict(title='Price [in $]',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      hovermode='closest')

        # create and plot figure
        figure = dict(data=[open_trace_original, open_trace_ma], layout=layout)
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.GOOGLE_STR, "exponential_smoothing_plot_{}.html".format(type)), auto_open=False)
        print(datetime.now(), ': exponential_smoothing_plot created.')


class Arima:

    def __init__(self, parser_object):
        self.parser_object = parser_object
        time_stamp_list = []
        close_list = []
        for single_google_recording in parser_object.single_google_recording_list:
            time_stamp_list.append(single_google_recording.time_stamp)
            close_list.append(single_google_recording.close)

        print(datetime.now(), ': Arima Model Received Data')
        self.series = close_list
        self.time_stamp_list = time_stamp_list

    def arimamodel(self, train, test, ar_param, order, ma_param, iterative=True):
        history = [x for x in train]
        preds = list()
        stamps = [int(len(test)/10)*i for i in range(1, 10)]
        if iterative:
            for i in range(len(test)):
                arima_model = ARIMA(history, order=(ar_param, order, ma_param))
                arima_model = arima_model.fit()
                output = arima_model.forecast()
                yhat = output[0]
                preds.append(yhat)
                obs = test[i]
                history.append(obs)
                if i in stamps:
                    print(str(datetime.now()) + ': Arima Prediction {}0% Complete: {} out of {}'.format((stamps.index(i)+1), i, len(test)))
        else:
            arima_model = ARIMA(history, order=(ar_param, order, ma_param))
            arima_model = arima_model.fit()
            preds = arima_model.predict(start=len(history), end=len(history)+len(test)-1)
        return preds

    def predict_and_plot(self, train_percentage=0.8, ar_param=1, order=1, ma_param=1, iterative=True):
        percentage = int(len(self.series)*train_percentage)
        train = self.series[:percentage]
        test = self.series[percentage:]
        predict = self.arimamodel(train, test, ar_param, order, ma_param, iterative)

        open_trace_original = go.Scattergl(x=self.time_stamp_list[percentage:], y=self.series[percentage:], mode='lines',
                                           name='original',
                                           opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')
        open_trace_ma = go.Scattergl(x=self.time_stamp_list[percentage:], y=predict, mode='lines',
                                     name='arima of p:{}, d:{}, q:{}'.format(ar_param, order, ma_param),
                                     opacity=1, showlegend=True, hoverinfo='text', legendgroup='lines')
        # design layout
        layout = dict(title='Arima',
                      xaxis=dict(title='Time',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      yaxis=dict(title='Price [in $]',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      hovermode='closest')

        # create and plot figure
        figure = dict(data=[open_trace_original, open_trace_ma], layout=layout)
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.GOOGLE_STR, "arima_plot_({},{},{}).html".format(ar_param, order, ma_param)), auto_open=False)
        print(datetime.now(), ': arima plot created.')

        return predict

class SupportVectorRegression():

    def __init__(self, kernel, degree, C, parser_object,
                 moving_average_object=None, kalman_filter_object=None, exp_smoothing_object=None):
        self.parser_object = parser_object
        self.moving_average_object = moving_average_object
        self.kalman_filter_object = kalman_filter_object
        self.exp_smoothing_object = exp_smoothing_object
        self.svr_model = SVR(kernel=kernel, degree=degree, C=C)

    def prepare_for_training(self, train_test_split):

        # extract recording list of respective stock
        if self.parser_object.name == GlobalConfig.BITCOIN_STR:
            if not self.moving_average_object == None:
                time_series_list = self.moving_average_object.moving_average_data_dict.\
                    get(GlobalConfig.MOVING_AVG_STR).get(self.moving_average_object.time_series)
                time_stamp_list = self.moving_average_object.moving_average_data_dict.\
                    get(GlobalConfig.MOVING_AVG_STR).get(GlobalConfig.TIMESTAMP_STR)
            elif not self.kalman_filter_object == None:
                time_series_list = self.kalman_filter_object.kalman_filter_dict. \
                    get(GlobalConfig.KALMAN_FILTER).get(self.kalman_filter_object.time_series)
                time_stamp_list = self.kalman_filter_object.kalman_filter_dict. \
                    get(GlobalConfig.KALMAN_FILTER).get(GlobalConfig.TIMESTAMP_STR)
            elif not self.exp_smoothing_object == None:
                time_series_list = self.exp_smoothing_object.exponential_smoothing_dict. \
                    get(GlobalConfig.EXPONENTIAL_SMOOTHING).get(self.exp_smoothing_object.time_series)
                time_stamp_list = self.exp_smoothing_object.exponential_smoothing_dict. \
                    get(GlobalConfig.EXPONENTIAL_SMOOTHING).get(GlobalConfig.TIMESTAMP_STR)
            else:
                recording_list = self.parser_object.single_bitcoin_recording_list
        elif self.parser_object.name == GlobalConfig.GOOGLE_STR:
            if not self.moving_average_object == None:
                time_series_list = self.moving_average_object.moving_average_data_dict. \
                    get(GlobalConfig.MOVING_AVG_STR).get(self.moving_average_object.time_series)
                time_stamp_list = self.moving_average_object.moving_average_data_dict. \
                    get(GlobalConfig.MOVING_AVG_STR).get(GlobalConfig.TIMESTAMP_STR)
            elif not self.kalman_filter_object == None:
                time_series_list = self.kalman_filter_object.kalman_filter_dict. \
                    get(GlobalConfig.KALMAN_FILTER).get(self.kalman_filter_object.time_series)
                time_stamp_list = self.kalman_filter_object.kalman_filter_dict. \
                    get(GlobalConfig.KALMAN_FILTER).get(GlobalConfig.TIMESTAMP_STR)
            elif not self.exp_smoothing_object == None:
                time_series_list = self.exp_smoothing_object.exponential_smoothing_dict. \
                    get(GlobalConfig.EXPONENTIAL_SMOOTHING).get(self.exp_smoothing_object.time_series)
                time_stamp_list = self.exp_smoothing_object.exponential_smoothing_dict. \
                    get(GlobalConfig.EXPONENTIAL_SMOOTHING).get(GlobalConfig.TIMESTAMP_STR)
            else:
                recording_list = self.parser_object.single_google_recording_list
        else:
            print('Valid parameters for <stock> are "Bitcoin" and "Google".')
            sys.exit(0)


        if (self.moving_average_object == None) and (self.kalman_filter_object == None) and (self.exp_smoothing_object == None):
            idx_list = []
            time_stamp_list = []
            close_list = []
            for idx, single_recording in enumerate(recording_list):
                idx_list.append(idx)
                time_stamp_list.append(single_recording.time_stamp)
                close_list.append(single_recording.close)

        else:
            idx_list = list(range(0, len(time_stamp_list), 1))
            time_stamp_list = time_stamp_list
            close_list = time_series_list


        features = np.expand_dims(np.asarray(idx_list), axis=1)
        labels = np.asarray(close_list)

        split_index = int(len(labels[:10000])*train_test_split)
        self.train_dates = time_stamp_list[:10000][:split_index]
        self.test_dates = time_stamp_list[:10000][split_index:]
        self.X_train = features[:10000][:split_index]
        self.y_train = labels[:10000][:split_index]
        self.X_test = features[:10000][split_index:]
        self.y_test = labels[:10000][split_index:]


    def train_model(self):
        self.prepare_for_training(train_test_split=0.75)
        self.svr_model.fit(self.X_train, self.y_train)
        print(datetime.now(), ': ', 'SVR training completed')


    def test_model(self):
        self.train_predictions = self.svr_model.predict(self.X_train)
        self.test_predictions = self.svr_model.predict(self.X_test)
        self.predictions = np.concatenate((self.train_predictions, self.test_predictions), axis=0)

        # get error metrics
        error_analyzer = ErrorAnalyzer(original=self.y_test, forecast=self.test_predictions)
        print('RMSE: ', error_analyzer.get_rmse())
        print('MAE: ', error_analyzer.get_mae())
        print('Correlation: ', error_analyzer.get_correlation())
        print('Momentum Error: ', error_analyzer.get_momentum_error())



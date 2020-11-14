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
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

class SimpleExponentialSmoothing:

    def __init__(self, parser_object):
        self.parser_object = parser_object
        self.ALPHA_MIN = 0.01
        self.ALPHA_MAX = 0.999
        self.ALPHA_STEP = 0.01
        self.horizon = 1
        time_stamp_list = []
        close_list = []
        for single_google_recording in parser_object.single_google_recording_list:
            time_stamp_list.append(single_google_recording.time_stamp)
            close_list.append(single_google_recording.close)

        print(datetime.now(), ': Exponential Smoothing Model Received Data')
        self.series = close_list

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
        trend = self.initial_trend(series, t)
        seasonals = self.initial_seasonal_components(series, t)
        seasonals.append(seasonals[0]) # To make the seasonals elements align with series elements
        for n in range(1, len(series)+horizon-1):
            if n >= len(series): # we are forecasting
                m = n - len(series) + 2
                result.append(smooth + m*trend + seasonals[n+1])
            else:
                val = series[n]
                last_smooth, smooth = smooth, alpha*(val-seasonals[n]) + (1-alpha)*(smooth+trend)
                trend = beta * (smooth-last_smooth) + (1-beta)*trend
                seasonals.append(gamma*(val-smooth) + (1-gamma)*seasonals[n])
                result.append(smooth + trend + seasonals[n+1])

        return result[len(series):len(series)+horizon]

    def initial_seasonal_components(self, series, t):
        seasonals = []
        season_averages = []
        n_seasons = int(len(series)/t)
        for j in range(n_seasons):
            season_averages.append(sum(series[t*j:t*j+t])/float(t))
        for i in range(t):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += series[t*j+i]-season_averages[j]
            seasonals.append(sum_of_vals_over_avg/n_seasons)
        return seasonals

    def initial_trend(self, series, t):
        sum = 0.0
        for i in range(t):
            sum += float(series[i+t] - series[i]) / t
        return sum / t

    def predict(self, seasons=4, alpha=0.5, beta=0.5, gamma=0.5, type="single"):
        preds = []
        for i in range(1, len(self.series)+1):
            if type == "single":
                preds.append(self.single_exponential_smoothing(self.series[:i], 1, alpha))
            elif type == "double" and i > 1:
                preds.append(self.double_exponential_smoothing(self.series[:i], 1, alpha, beta))
            elif type == "triple":
                preds.append(self.triple_exponential_smoothing(self.series[:i], seasons, 1, alpha, beta, gamma))
        print("Successful prediction for {}".format(type))
        return preds

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

    def parameter_tune(self, train_size):
        x = self.series
        size = int(len(x) * train_size)
        train, test = x[0:size], x[size:len(x)]
        history = [x for x in train]
        predictions = []
        for t in range(len(test)):
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print("Predicted-%f expected-%f" % (yhat, obs))
        error = mean_squared_error(test, predictions)
        print("Test MSE: %.3f" % error)

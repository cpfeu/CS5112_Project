import os
from datetime import datetime
import plotly.offline as po
import plotly.graph_objs as go
from global_config import GlobalConfig


class BitcoinVisualizer:

    def __init__(self, parser_object, preprocessor_object):
        self.parser_object = parser_object
        self.preprocessor_object = preprocessor_object

    def plot_all_in_one_chart(self):
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
        open_trace = go.Scattergl(x=time_stamp_list, y=open_list, mode='lines+markers', name=GlobalConfig.OPEN_STR)
        close_trace = go.Scattergl(x=time_stamp_list, y=close_list, mode='lines+markers', name=GlobalConfig.CLOSE_STR)
        high_trace = go.Scattergl(x=time_stamp_list, y=high_list, mode='lines+markers', name=GlobalConfig.HIGH_STR)
        low_trace = go.Scattergl(x=time_stamp_list, y=low_list, mode='lines+markers', name=GlobalConfig.LOW_STR)
        volume_trace = go.Scattergl(x=time_stamp_list, y=volume_list, mode='lines+markers', name=GlobalConfig.VOLUME_STR)

        # create and plot figure
        figure = dict(data=[open_trace, close_trace, high_trace, low_trace, volume_trace])
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.BITCOIN_STR, "All_in_one_plot.html"), auto_open=False)
        print(datetime.now(), ': all_in_one_plot created.')


    def plot_moving_average(self):

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





class GoogleVisualizer:

    def __init__(self, parser_object, preprocessor_object):
        self.parser_object = parser_object
        self.preprocessor_object = preprocessor_object

    def plot_all_in_one_chart(self):
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
        open_trace = go.Scattergl(x=time_stamp_list, y=open_list, mode='lines+markers', name=GlobalConfig.OPEN_STR)
        close_trace = go.Scattergl(x=time_stamp_list, y=close_list, mode='lines+markers', name=GlobalConfig.CLOSE_STR)
        high_trace = go.Scattergl(x=time_stamp_list, y=high_list, mode='lines+markers', name=GlobalConfig.HIGH_STR)
        low_trace = go.Scattergl(x=time_stamp_list, y=low_list, mode='lines+markers', name=GlobalConfig.LOW_STR)
        volume_trace = go.Scattergl(x=time_stamp_list, y=volume_list, mode='lines+markers', name=GlobalConfig.VOLUME_STR)

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





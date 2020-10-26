import os
from datetime import datetime
import plotly.offline as po
import plotly.graph_objs as go
from global_config import GlobalConfig


class BitcoinVisualizer:

    def __init__(self, parser_object):
        self.parser_object = parser_object

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
        open_trace = go.Scattergl(x=time_stamp_list, y=open_list, mode='lines+markers', name='open')
        close_trace = go.Scattergl(x=time_stamp_list, y=close_list, mode='lines+markers', name='close')
        high_trace = go.Scattergl(x=time_stamp_list, y=high_list, mode='lines+markers', name='high')
        low_trace = go.Scattergl(x=time_stamp_list, y=low_list, mode='lines+markers', name='low')
        volume_trace = go.Scattergl(x=time_stamp_list, y=volume_list, mode='lines+markers', name='volume')

        # create and plot figure
        figure = dict(data=[open_trace, close_trace, high_trace, low_trace, volume_trace])
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.BITCOIN_STR, "All_in_one_plot.html"), auto_open=False)
        print(datetime.now(), ': all_in_one_plot created.')


class GoogleVisualizer:

    def __init__(self, parser_object):
        self.parser_object = parser_object

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
        open_trace = go.Scattergl(x=time_stamp_list, y=open_list, mode='lines+markers', name='open')
        close_trace = go.Scattergl(x=time_stamp_list, y=close_list, mode='lines+markers', name='close')
        high_trace = go.Scattergl(x=time_stamp_list, y=high_list, mode='lines+markers', name='high')
        low_trace = go.Scattergl(x=time_stamp_list, y=low_list, mode='lines+markers', name='low')
        volume_trace = go.Scattergl(x=time_stamp_list, y=volume_list, mode='lines+markers', name='volume')

        # create and plot figure
        figure = dict(data=[open_trace, close_trace, high_trace, low_trace, volume_trace])
        if self.parser_object.data_path == GlobalConfig.GOOGLE_DATA_PATH:
            filename = 'All_in_one_plot.html'
        else:
            filename = 'All_in_one_plot_extended.html'
        po.plot(figure, filename=os.path.join(GlobalConfig.WORKING_DIR_PATH,
                                              GlobalConfig.GOOGLE_STR, filename), auto_open=False)
        print(datetime.now(), ': all_in_one_plot created.')







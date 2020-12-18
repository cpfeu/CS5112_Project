import os
import math
import pandas as pd
from datetime import datetime
from global_config import GlobalConfig


class BitcoinParser:

    def __init__(self, data_path=GlobalConfig.BITCOIN_DATA_PATH):

        '''
        A <BitcoinParser> object holds the Bitcoin name and the data path
        :param data_path: <str> - specifies where Bitcoin data is located
        '''

        self.name = GlobalConfig.BITCOIN_STR
        self.data_path = data_path
        os.makedirs(os.path.join(GlobalConfig.WORKING_DIR_PATH, GlobalConfig.BITCOIN_STR), exist_ok=True)

    class SingleBitcoinRecording:

        '''
        A <SingleBitcoinRecording> object holds a name, a timestamp, the open, close, high and low price and
        the trading volume. Each <SingleBitcoinRecording> represents 1 minute.
        '''

        def __init__(self, name, time_stamp, open, high, low, volume, close):
            self.name = name
            self.time_stamp = time_stamp
            self.open = open
            self.high = high
            self.low = low
            self.volume = volume
            self.close = close

    def parse_bitcoin_data(self):

        '''
        The Bitcoin data is parsed and stored in different formats, including a Pandas DataFrame, a dictionary and
        a list of <SingleBitcoinRecording> objects. This list is the most important one and is used
        by different objects throughout the project.
        :return:
        '''

        self.data_pd = pd.read_csv(filepath_or_buffer=self.data_path, sep=',', header=1, index_col=False)
        self.data_dict = self.data_pd.to_dict(orient='list')
        self.single_bitcoin_recording_list = []
        for ts_idx, time_stamp in enumerate(self.data_dict.get(GlobalConfig.BITCOIN_TIME_STAMP_NAME_STR)):
            if not math.isnan(self.data_dict.get(GlobalConfig.BITCOIN_OPEN_STR)[ts_idx]):
                open_val = self.data_dict.get(GlobalConfig.BITCOIN_OPEN_STR)[ts_idx]
                close_val = self.data_dict.get(GlobalConfig.BITCOIN_CLOSE_STR)[ts_idx]
                high_val = self.data_dict.get(GlobalConfig.BITCOIN_HIGH_STR)[ts_idx]
                low_val = self.data_dict.get(GlobalConfig.BITCOIN_LOW_STR)[ts_idx]
                volume_val = self.data_dict.get(GlobalConfig.BITCOIN_VOLUME_STR)[ts_idx]
                self.single_bitcoin_recording_list.append(self.SingleBitcoinRecording(name='Bitcoin_USD',
                                                                                      time_stamp=datetime.utcfromtimestamp(time_stamp / 1000),
                                                                                      open=open_val,
                                                                                      high=high_val,
                                                                                      low=low_val,
                                                                                      volume=volume_val,
                                                                                      close=close_val))
        print(datetime.now(), ': Bitcoin parsing completed.')




class GoogleParser:

    def __init__(self, data_path=GlobalConfig.GOOGLE_DATA_PATH):

        '''
        A <GoogleParser> object holds the Google name and the data path
        :param data_path: <str> - specifies where Google data is located
        '''

        self.name = GlobalConfig.GOOGLE_STR
        self.data_path = data_path
        os.makedirs(os.path.join(GlobalConfig.WORKING_DIR_PATH, GlobalConfig.GOOGLE_STR), exist_ok=True)

    class SingleGoogleRecording:

        '''
        A <SingleGoogleRecording> object holds a name, a timestamp, the open, close, high and low price and
        the trading volume. Each <SingleGoogleRecording> represents one 'interval' that was passed into a
         <DataPuller> object.
        '''

        def __init__(self, name, time_stamp, open, high, low, volume, close):
            self.name = name
            self.time_stamp = time_stamp
            self.open = open
            self.high = high
            self.low = low
            self.volume = volume
            self.close = close

    def parse_google_data(self):

        '''
        The Google data is parsed and stored in different formats, including a Pandas DataFrame, a dictionary and
        a list of <SingleGoogleRecording> objects. This list is the most important one and is used
        by different objects throughout the project.
        :return:
        '''

        self.data_pd = pd.read_csv(filepath_or_buffer=self.data_path, sep=',', header=0, index_col=False)
        column_names = self.data_pd.columns
        self.data_dict = self.data_pd.to_dict(orient='list')
        self.single_google_recording_list = []
        for ts_idx, time_stamp in enumerate(self.data_dict.get(list(column_names)[0])):
            if not math.isnan(self.data_dict.get(column_names[1])[ts_idx]):
                open_val = self.data_dict.get(column_names[1])[ts_idx]
                close_val = self.data_dict.get(column_names[2])[ts_idx]
                high_val = self.data_dict.get(column_names[3])[ts_idx]
                low_val = self.data_dict.get(column_names[4])[ts_idx]
                volume_val = self.data_dict.get(column_names[5])[ts_idx]
                self.single_google_recording_list.append(self.SingleGoogleRecording(name='Google_USD',
                                                                                      time_stamp=time_stamp,
                                                                                      open=open_val,
                                                                                      high=high_val,
                                                                                      low=low_val,
                                                                                      volume=volume_val,
                                                                                      close=close_val))
        print(datetime.now(), ': Google parsing completed.')









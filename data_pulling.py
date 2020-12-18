import os
import csv
import requests
from datetime import datetime
from global_config import GlobalConfig
from alpha_vantage.timeseries import TimeSeries


class DataPuller:

    def __init__(self, api_key=GlobalConfig.ALPHA_VANTAGE_API_KEY, ticker='GOOGL', interval='1min'):

        '''
        :param api_key: <str> - used to make a request at AlphaVantage
        :param ticker: <str> - the respective stock ticker of the stock exchange
        :param interval: <str> - detail of data set, e.g. '1min' means that data from every minute is pulled
        '''

        self.api_key = api_key
        self.ticker = ticker
        self.interval = interval

    def pull_data(self):

        '''
        A function that uses the variables of it's respective <DataPuller>-object.
        There are two API keys that can be used: One API key for extracting data of the last month and
        another API to extract data of the last two years. In both cases, the downloaded data set is stored
        in a csv-file in the data_base_path directory specified in the <LocalConfig>-class
        :return:
        '''

        # pull normal dataset (about 2 weeks of trading)
        if self.api_key == GlobalConfig.ALPHA_VANTAGE_API_KEY:
            time_series = TimeSeries(key=self.api_key, output_format='pandas')
            data = time_series.get_intraday(symbol=self.ticker, interval=self.interval, outputsize='full')

            data_pd = data[0]
            data_information = data[1]
            data_pd.to_csv(path_or_buf=os.path.join(GlobalConfig.BASE_DATA_PATH,
                                                    self.ticker + '_' + self.interval + '.csv'),
                           sep=',', header=data_pd.columns)

            print(datetime.now(), ':', self.ticker, 'successfully pulled and stored.')


        # pull extended dataset (about 2 years of trading)
        elif self.api_key == GlobalConfig.ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY:
            data_batch_list = []
            for slice in GlobalConfig.SLICE_LIST:
                data = requests.get('https://www.alphavantage.co/query?'
                                    'function=TIME_SERIES_INTRADAY_EXTENDED&'
                                    'symbol={}&interval={}&'
                                    'slice={}&apikey={}'.format(self.ticker, self.interval, slice, self.api_key))
                data_content = data.content
                data_content_str = data_content.decode('utf-8')
                data_batch_list.append(data_content_str)
                print('Pulled data for', self.ticker, 'at', slice)

            # store in csv file
            with open(os.path.join(GlobalConfig.BASE_DATA_PATH,
                                   self.ticker+'_'+self.interval+'_extended_history.csv'), 'w', newline='') as file:
                csv_writer = csv.writer(file, delimiter=',')
                for idx, data_batch in enumerate(data_batch_list):
                    if idx == 0:
                        for line in data_batch.split('\r\n'):
                            csv_writer.writerow(line.split(','))
                    else:
                        for idx, line in enumerate(data_batch.split('\r\n')):
                            if idx == 0:
                                continue
                            else:
                                csv_writer.writerow(line.split(','))
                print(datetime.now(), ':', self.ticker, 'successfully pulled and stored.')


        # no data can be pulled since key is not valid
        else:
            print('Unknown API key.')






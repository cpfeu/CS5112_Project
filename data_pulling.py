import os
from datetime import datetime
from global_config import GlobalConfig
from alpha_vantage.timeseries import TimeSeries

class DataPuller:

    def __init__(self, api_key=GlobalConfig.ALPHA_VANTAGE_API_KEY, ticker='GOOGL', interval='1min'):
        self.api_key = api_key
        self.ticker = ticker
        self.interval = interval

    def pull_data(self):
        time_series = TimeSeries(key=self.api_key, output_format='pandas')
        data = time_series.get_intraday(symbol=self.ticker, interval=self.interval, outputsize='full')
        data_pd = data[0]
        data_information = data[1]
        data_pd.to_csv(path_or_buf=os.path.join(GlobalConfig.BASE_DATA_PATH,
                                                self.ticker+'_'+self.interval+'.csv'),
                       sep=',', header=data_pd.columns)

        print(datetime.now(), ':', self.ticker, 'successfully pulled and stored.')

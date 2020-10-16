from datetime import datetime
from global_config import GlobalConfig

from data_pulling import DataPuller
from data_parsing import BitcoinParser, GoogleParser
from data_visualization import BitcoinVisualizer, GoogleVisualizer


if __name__ == '__main__':

    # starting time
    starting_time = datetime.now()
    print(starting_time, ': Program started.')

    #==========commands==========

    # pull data
    data_puller = DataPuller(api_key=GlobalConfig.ALPHA_VANTAGE_API_KEY, ticker='GOOGL', interval='1min')
    data_puller.pull_data()

    # parse data
    # bitcoin_parser = BitcoinParser()
    # bitcoin_parser.parse_bitcoin_data()
    google_parser = GoogleParser()
    google_parser.parse_google_data()

    # visualize data
    # bitcoin_visualizer = BitcoinVisualizer(bitcoin_parser)
    # bitcoin_visualizer.plot_all_in_one_chart()
    google_visualizer = GoogleVisualizer(google_parser)
    google_visualizer.plot_all_in_one_chart()


    # ending time
    ending_time=datetime.now()
    print(ending_time, ': Program finished.')

    # execution length
    print('Program took:', ending_time-starting_time, 'to run.')

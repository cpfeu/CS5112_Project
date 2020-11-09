from datetime import datetime
from global_config import GlobalConfig

from data_pulling import DataPuller
from data_parsing import BitcoinParser, GoogleParser
from data_preprocessing import MovingAverage, KalmanFilter
from data_visualization import BitcoinVisualizer, GoogleVisualizer


if __name__ == '__main__':

    # starting time
    starting_time = datetime.now()
    print(starting_time, ': Program started.')

    #==========commands==========

    # pull data
    # data_puller = DataPuller(api_key=GlobalConfig.ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY,
    #                          ticker='GOOGL',
    #                          interval='1min')
    # data_puller.pull_data()

    # parse data
    bitcoin_parser = BitcoinParser()
    bitcoin_parser.parse_bitcoin_data()
    #google_parser = GoogleParser(data_path=GlobalConfig.GOOGLE_DATA_EXTENDED_PATH)
    #google_parser.parse_google_data()

    # preprocess data
    #bitcoin_ma = MovingAverage(parser_object=bitcoin_parser,
    #                           time_series=GlobalConfig.CLOSE_STR,
    #                           window_size=7501,
    #                           weighted=False,
    #                           weights=[0.1, 0.2, 0.3, 0.4])
    #bitcoin_ma.calculate_moving_average()
    bitcoin_kalman = KalmanFilter(parser_object=bitcoin_parser,
                                  time_series=GlobalConfig.CLOSE_STR,
                                  Q=1e-5, R=0.1**2, prediction_time=1000)
    bitcoin_kalman.calculate_kalman_filter()

    #google_ma = MovingAverage(parser_object=google_parser,
    #                          time_series=GlobalConfig.CLOSE_STR,
    #                          window_size=7501,
    #                          weighted=False,
    #                          weights=[0.1, 0.2, 0.3, 0.4])
    #google_ma.calculate_moving_average()

    # visualize data
    # bitcoin_visualizer = BitcoinVisualizer(bitcoin_parser, bitcoin_ma)
    bitcoin_visualizer = BitcoinVisualizer(bitcoin_parser, bitcoin_kalman)
    # bitcoin_visualizer.plot_all_in_one_chart()
    #bitcoin_visualizer.plot_moving_average()
    bitcoin_visualizer.plot_kalman_filter()
    #google_visualizer = GoogleVisualizer(google_parser, google_ma)
    # google_visualizer.plot_all_in_one_chart()
    #google_visualizer.plot_moving_average()


    # ending time
    ending_time=datetime.now()
    print(ending_time, ': Program finished.')

    # execution length
    print('Program took:', ending_time-starting_time, 'to run.')





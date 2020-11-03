from datetime import datetime
from global_config import GlobalConfig

from data_pulling import DataPuller
from data_parsing import BitcoinParser, GoogleParser
from data_preprocessing import PreprocessorObject
from data_visualization import BitcoinVisualizer, GoogleVisualizer
from data_forecasting import SimpleExponentialSmoothing


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
    # google_parser = GoogleParser(data_path=GlobalConfig.GOOGLE_DATA_EXTENDED_PATH)
    # google_parser.parse_google_data()

    # preprocess data
    bitcoin_preprocessor = PreprocessorObject(parser_object=bitcoin_parser)
    bitcoin_preprocessor.moving_average(stock=GlobalConfig.BITCOIN_STR,
                                        time_series=GlobalConfig.HIGH_STR,
                                        window_size=5, weighted=True, weights=[0.2, 0.2, 0.2, 0.2, 0.2])
    # google_preprocessor = PreprocessorObject(parser_object=google_parser)
    # google_preprocessor.moving_average(stock=GlobalConfig.GOOGLE_STR,
    #                                     time_series=GlobalConfig.HIGH_STR,
    #                                     window_size=10001, weighted=False, weights=[])

    # visualize data
    bitcoin_visualizer = BitcoinVisualizer(bitcoin_parser, bitcoin_preprocessor)
    # bitcoin_visualizer.plot_all_in_one_chart()
    bitcoin_visualizer.plot_moving_average(time_series=GlobalConfig.HIGH_STR)
    # google_visualizer = GoogleVisualizer(google_parser, google_preprocessor)
    # # google_visualizer.plot_all_in_one_chart()
    # google_visualizer.plot_moving_average(time_series=GlobalConfig.HIGH_STR)

    # exponential smoothing model
    exponential_smoothing_model = SimpleExponentialSmoothing(google_parser)
    exponential_smoothing_model.initialize_model(horizon=7, train_test_split=0.5)
    exponential_smoothing_model.predict()

    # ending time
    ending_time=datetime.now()
    print(ending_time, ': Program finished.')

    # execution length
    print('Program took:', ending_time-starting_time, 'to run.')





from datetime import datetime
from global_config import GlobalConfig

from data_pulling import DataPuller
from data_parsing import BitcoinParser, GoogleParser
from data_preprocessing import MovingAverage, KalmanFilter, DecomposeTimeSeries, ExponentialSmoothing
from data_visualization import BitcoinVisualizer, GoogleVisualizer
from data_forecasting import SimpleExponentialSmoothingForecaster, Arima, SupportVectorRegression


if __name__ == '__main__':

    # starting time
    starting_time = datetime.now()
    print(starting_time, ': Program started.')


    # =============== pull data ===============
    #data_puller = DataPuller(api_key=GlobalConfig.ALPHA_VANTAGE_API_KEY,
    #                           ticker='GOOGL',
    #                           interval='1min')
    #data_puller.pull_data()

    # =============== parse data ===============
    # bitcoin_parser = BitcoinParser()
    # bitcoin_parser.parse_bitcoin_data()
    google_parser = GoogleParser(data_path=GlobalConfig.GOOGLE_DATA_PATH)
    google_parser.parse_google_data()

    # =============== preprocess data ===============
    # bitcoin_ma = MovingAverage(parser_object=bitcoin_parser,
    #                            time_series=GlobalConfig.HIGH_STR,
    #                            window_size=7501,
    #                            weighted=False,
    #                            weights=[0.1, 0.2, 0.3, 0.4])
    # bitcoin_ma.calculate_moving_average()
    # bitcoin_kalman = KalmanFilter(parser_object=bitcoin_parser,
    #                               time_series=GlobalConfig.HIGH_STR,
    #                               Q=1e-5, R=0.1**2, prediction_time=360)
    # bitcoin_kalman.calculate_kalman_filter()
    # bitcoin_dec = DecomposeTimeSeries(parser_object=bitcoin_parser,
    #                                   time_series=GlobalConfig.CLOSE_STR,
    #                                   decompose_model=GlobalConfig.ADDITIVE_DECOMPOSITION,
    #                                   period=28)
    # bitcoin_dec.decompose_time_series(decompose_with_kalman_filter=True, show_decomposed_ts=True)
    # google_ma = MovingAverage(parser_object=google_parser,
    #                           time_series=GlobalConfig.HIGH_STR,
    #                           window_size=7501,
    #                           weighted=False,
    #                           weights=[0.1, 0.2, 0.3, 0.4])
    # google_ma.calculate_moving_average()
    # google_kalman = KalmanFilter(parser_object=google_parser,
    #                              time_series=GlobalConfig.HIGH_STR,
    #                              Q=1e-5, R=0.1**2, prediction_time=360)
    # google_kalman.calculate_kalman_filter()
    # google_dec = DecomposeTimeSeries(parser_object=google_parser,
    #                                  time_series=GlobalConfig.CLOSE_STR,
    #                                  decompose_model=GlobalConfig.ADDITIVE_DECOMPOSITION,
    #                                  period=28)
    # google_dec.decompose_time_series(decompose_with_kalman_filter=True, show_decomposed_ts=True)

    # =============== forecasting ===============
    # bitcoin_svr = SupportVectorRegression(kernel='rbf', degree=12, C=1,
    #                                       parser_object=bitcoin_parser,
    #                                       moving_average_object=None,
    #                                       kalman_filter_object=None)
    # bitcoin_svr.train_model()
    # bitcoin_svr.test_model()
    # google_svr = SupportVectorRegression(kernel='rbf', degree=12, C=1,
    #                                      parser_object=google_parser,
    #                                      moving_average_object=None,
    #                                      kalman_filter_object=None)
    # google_svr.train_model()
    # google_svr.test_model()

    # =============== visualize data ===============
    # bitcoin_visualizer_1 = BitcoinVisualizer(bitcoin_parser, None)
    # bitcoin_visualizer_2 = BitcoinVisualizer(bitcoin_parser, bitcoin_kalman)
    # bitcoin_visualizer_3 = BitcoinVisualizer(bitcoin_parser)
    # bitcoin_visualizer_4 = BitcoinVisualizer(bitcoin_parser, forecaster_object=bitcoin_svr)
    # bitcoin_visualizer_1.plot_all_in_one_chart()
    # bitcoin_visualizer_1.plot_moving_average()
    # bitcoin_visualizer_2.plot_kalman_filter()
    # bitcoin_visualizer_3.plot_autocorrelation(lags=500)
    # bitcoin_visualizer_4.plot_svr_performace()
    # google_visualizer_1 = GoogleVisualizer(google_parser, None)
    # google_visualizer_2 = GoogleVisualizer(google_parser, google_kalman)
    # google_visualizer_3 = GoogleVisualizer(google_parser)
    # google_visualizer_4 = GoogleVisualizer(google_parser, forecaster_object=google_svr)
    # google_visualizer_1.plot_all_in_one_chart()
    # google_visualizer_1.plot_moving_average()
    # google_visualizer_2.plot_kalman_filter()
    # google_visualizer_3.plot_autocorrelation(lags=500)
    # google_kalman.calculate_kalman_filter()
    # google_visualizer_4.plot_svr_performace()

    google_exp_smth = ExponentialSmoothing(parser_object=google_parser,
                                           time_series=GlobalConfig.CLOSE_STR)
    print('Single_Smoothing')
    google_exp_smth.calculate_single_exponential_smoothing()
    print('Double_Smoothing')
    google_exp_smth.calculate_double_exponential_smoothing()

    # =============== exponential smoothing model ===============
    # exponential_smoothing_model = SimpleExponentialSmoothingForecaster(time_series=google_ma.moving_average_data_dict[GlobalConfig.MOVING_AVG_STR][GlobalConfig.CLOSE_STR],
    #                                                        time_stamps=google_ma.moving_average_data_dict[GlobalConfig.MOVING_AVG_STR][GlobalConfig.TIMESTAMP_STR])
    # exponential_smoothing_model.predict(type="single")
    # exponential_smoothing_model.predict(type="double")
    # exponential_smoothing_model.predict(type="triple")
    # arima = Arima(google_parser)
    # arima.predict_and_plot()

    # ending time
    ending_time=datetime.now()
    print(ending_time, ': Program finished.')

    # execution length
    print('Program took:', ending_time-starting_time, 'to run.')

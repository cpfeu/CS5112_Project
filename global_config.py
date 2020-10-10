from local_config import LocalConfig

class GlobalConfig:

    NUM_CPUs = LocalConfig.NUM_CPUs
    ALPHA_VANTAGE_API_KEY = LocalConfig.ALPHA_VANTAGE_API_KEY
    WORKING_DIR_PATH = LocalConfig.WORKING_DIR_PATH
    BASE_DATA_PATH = LocalConfig.BASE_DATA_PATH

    # Bitcoin parameters
    BITCOIN_DATA_PATH = LocalConfig.BITCOIN_DATA_PATH
    BITCOIN_STR = 'Bitcoin'
    BITCOIN_SKIP_NAMES_LIST = ['Unix Timestamp', 'Symbol', 'Date']
    BITCOIN_TIME_STAMP_NAME_STR = 'Unix Timestamp'
    BITCOIN_OPEN_STR = 'Open'
    BITCOIN_HIGH_STR = 'High'
    BITCOIN_LOW_STR = 'Low'
    BITCOIN_CLOSE_STR = 'Close'
    BITCOIN_VOLUME_STR = 'Volume'

    # Google parameters
    GOOGLE_DATA_PATH = LocalConfig.GOOGLE_DATA_PATH
    GOOGLE_STR = 'Google'
    GOOGLE_TIME_STAMP_NAME_STR = 'date'
    GOOGLE_OPEN_STR = '1. open'
    GOOGLE_HIGH_STR = '2. high'
    GOOGLE_LOW_STR = '3. low'
    GOOGLE_CLOSE_STR = '4. close'
    GOOGLE_VOLUME_STR = '5. volume'
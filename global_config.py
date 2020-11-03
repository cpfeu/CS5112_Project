from local_config import LocalConfig

class GlobalConfig:

    NUM_CPUs = LocalConfig.NUM_CPUs
    ALPHA_VANTAGE_API_KEY = LocalConfig.ALPHA_VANTAGE_API_KEY
    ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY = LocalConfig.ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY
    WORKING_DIR_PATH = LocalConfig.WORKING_DIR_PATH
    BASE_DATA_PATH = LocalConfig.BASE_DATA_PATH
    SLICE_LIST = ['year1month1', 'year1month2', 'year1month3', 'year1month4', 'year1month5', 'year1month6',
                  'year1month7', 'year1month8', 'year1month9', 'year1month10', 'year1month11', 'year1month12',
                  'year2month1', 'year2month2', 'year2month3', 'year2month4', 'year2month5', 'year2month6',
                  'year2month7', 'year2month8', 'year2month9', 'year2month10', 'year2month11', 'year2month12']

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
    GOOGLE_DATA_EXTENDED_PATH = LocalConfig.GOOGLE_DATA_PATH_EXTENDED
    GOOGLE_STR = 'Google'
    GOOGLE_TIME_STAMP_NAME_STR = 'date'

    # Time series names in single recording object
    OPEN_STR = 'open'
    HIGH_STR = 'high'
    LOW_STR = 'low'
    CLOSE_STR = 'close'
    VOLUME_STR = 'volume'

    # Method Strings
    MOVING_AVG_STR = 'Moving_Average'

    # Time stamp string
    TIMESTAMP_STR = 'Time_Stamp'




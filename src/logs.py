#import packages
from imports import *

# a function  to create and save logs in the log files
def log(file):
    if not os.path.isfile(file):
        open(file, "w+").close()

    console_logging_format = '%(levelname)s %(message)s'
    file_logging_format = '%(levelname)s: %(asctime)s: %(message)s'

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    # create a file handler for output file
    handler = logging.FileHandler(file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

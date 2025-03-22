import logging

class ColoredFormatter(logging.Formatter):
    COLOR_CODES = {
        'DEBUG': '\033[94m',  # blue
        'INFO': '\033[92m',   # green
        'WARNING': '\033[93m',  # yellow
        'ERROR': '\033[91m',  # red
        'CRITICAL': '\033[41m\033[97m'  # red background color and white text
    }
    RESET_CODE = '\033[0m'  # called to return to standard terminal text color

    def format(self, record):
        # Get the color corresponding to the log level
        color = self.COLOR_CODES.get(record.levelname, '')
        # Add color to log messages and reset color at the end
        formatted_message = f"{color}{super().format(record)}{self.RESET_CODE}"
        return formatted_message


console_msg_format = "%(asctime)s %(levelname)s: %(message)s"
logger = logging.getLogger()


def initialize_logger(log_level=logging.DEBUG, log_file='app.log'):
    # Create the root logger
    logger.setLevel(log_level)

    # Set up logging to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = ColoredFormatter(console_msg_format)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # Set up file logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter(console_msg_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


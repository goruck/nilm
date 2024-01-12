"""Logger for NILM project.

Copyright (c) 2023~2024 Lindo St. Angel
"""

import logging
import time

class Logger():
    """Logger class for NILM project."""
    def __init__(self, level:str='info', log_file_name:str=None, append:bool=False) -> None:
        """Inits logger."""
        if log_file_name is None:
            log_file_name = '{}.log'.format(time.strftime("%Y-%m-%d-%H:%M:%S").replace(':','-'))

        mode = 'a' if append else 'w'
        with open(log_file_name, mode=mode, encoding='utf-8'):
            pass

        self.root_logger = logging.getLogger()

        log_formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]  %(message)s')

        file_handler = logging.Handler('{0}'.format(log_file_name))
        file_handler.setFormatter(log_formatter)
        self.root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.root_logger.addHandler(console_handler)

        # Set lowest-severity log message logger will handle.
        if level == 'debug':
            self.root_logger.setLevel(logging.DEBUG)
        elif level == 'warning':
            self.root_logger.setLevel(logging.WARNING)
        elif level == 'critical':
            self.root_logger.setLevel(logging.CRITICAL)
        else:
            self.root_logger.setLevel(logging.INFO)

        # Disable debug messages from the following modules.
        disable_debug_modules = [
            'matplotlib',
            'matplotlib.font',
            'matplotlib.pyplot',
            'matplotlib.font_manager',
            'PIL'
        ]
        for module in disable_debug_modules:
            logger = logging.getLogger(module)
            logger.setLevel(logging.INFO)

    def log(self, string:str, level:str='info') -> None:
        """Log message per level."""
        if level == 'debug':
            self.root_logger.debug(string)
        elif level == 'warning':
            self.root_logger.warning(string)
        elif level == 'critical':
            self.root_logger.critical(string)
        else:
            self.root_logger.info(string)

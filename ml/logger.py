"""Logger for NILM project.

Copyright (c) 2023 Lindo St. Angel
"""

import logging
import time

class Logger():
    def __init__(self, log_file_name=None) -> None:
        if log_file_name is None:
            log_file_name = '{}.log'.format(time.strftime("%Y-%m-%d-%H:%M:%S").replace(':','-'))
        with open(log_file_name, 'w'):
            pass
        
        self.rootLogger = logging.getLogger()

        logFormatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]  %(message)s')

        fileHandler = logging.FileHandler('{0}'.format(log_file_name))
        fileHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(consoleHandler)

        self.rootLogger.setLevel(logging.DEBUG)

        # Disable debug messages from the following modules.
        disable_debug_modules = ['matplotlib',
                                 'matplotlib.font',
                                 'matplotlib.pyplot',
                                 'matplotlib.font_manager',
                                 'PIL']
        for module in disable_debug_modules:
            logger = logging.getLogger(module)
            logger.setLevel(logging.INFO)

    def log(self, string:str, level:str='info') -> None:
        if level == 'debug':
            self.rootLogger.debug(string)
        elif level == 'warning':
            self.rootLogger.warning(string)
        elif level == 'critical':
            self.rootLogger.critical(string)
        else:
            self.rootLogger.info(string)
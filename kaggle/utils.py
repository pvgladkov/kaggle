# coding: utf-8

import logging


def create_logger(logger_name):
    logger = logging.getLogger(logger_name)

    console_handler_name = '%s-%s' % (logger_name, 'console_handler')
    root_handler_name = 'root'
    handler_names = {handler.name for handler in logger.handlers}

    if {root_handler_name, console_handler_name} < set(handler_names):
        return logger

    logger.setLevel(logging.INFO)

    plain_line_formatter = logging.Formatter(
        u'[%(process)d][%(processName)s]%(filename)s[LINE:%(lineno)3d]# %(levelname)-8s [%(asctime)s] %(message)s'
    )

    if console_handler_name not in handler_names:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(plain_line_formatter)
        console_handler.set_name(console_handler_name)
        logger.addHandler(console_handler)

    return logger
import logging

def set_debug_log_level():
    setDebugLogLevel()
    logging.basicConfig(level=logging.DEBUG)


def set_info_log_level():
    setInfoLogLevel()
    logging.basicConfig(level=logging.INFO)


def set_warn_log_level():
    setWarnLogLevel()
    logging.basicConfig(level=logging.WARN)


def set_err_log_level():
    setErrLogLevel()
    logging.basicConfig(level=logging.ERROR)

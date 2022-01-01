# noinspection PyUnresolvedReferences
from .cpp_api import Tensor, Parameter, PythonKernel
from .cpp_api import MatMul, Relu, Sum, Log, Sigmoid, Softmax, Add, Mul, save_op
from .cpp_api import set_debug_log_level, set_info_log_level, set_warn_log_level, set_err_log_level
from .cpp_api import get_numpy_kernels, register_add_op, register_mul_op
from .numpy_kernels import *
from .utils import astensor


def set_debug_log_level():
    from .cpp_api import set_debug_log_level as set_debug_level
    import logging
    set_debug_level()
    logging.basicConfig(level=logging.DEBUG)


def init_logger():
    import logging
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            # logging.FileHandler("debug.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def windows_enable_ansi_terminal_mode():
    import sys
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        result = kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        if result == 0: raise Exception
        return True
    except:
        return False


init_logger()
windows_enable_ansi_terminal_mode()
register_add_op(Add)
register_mul_op(Mul)

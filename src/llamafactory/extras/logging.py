import logging
import os
import sys


def _rank0():
    return int(os.getenv("LOCAL_RANK", "0")) == 0


def get_logger(name=None):
    root_name = __name__.split(".")[0]
    root = logging.getLogger(root_name)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s", "%Y-%m-%d %H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(os.getenv("LLAMAFACTORY_VERBOSITY", "INFO").upper())
        root.propagate = False
    return logging.getLogger(name or root_name)


def _info_rank0(self, *args, **kwargs):
    if _rank0():
        self.info(*args, **kwargs)


def _warning_rank0(self, *args, **kwargs):
    if _rank0():
        self.warning(*args, **kwargs)


logging.Logger.info_rank0 = _info_rank0
logging.Logger.warning_rank0 = _warning_rank0
logging.Logger.warning_rank0_once = _warning_rank0

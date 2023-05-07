"""
Simple logger that supports writing to a log file
"""
import logging
import os
import sys
from pathlib import Path

class SimpleLogger(logging.getLoggerClass()):

    def __init__(self, name="Logger") -> None:
        super(SimpleLogger, self).__init__(name)
        self.setLevel(logging.DEBUG)
        self.log_format = logging.Formatter(
            fmt="%(asctime)s %(name)s:%(levelname)s\n%(message)s\n",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return

    """
    Context manager
    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return

    """
    Public
    """

    def set_stream_handler(self, level=logging.DEBUG, stream=sys.stdout) -> None:
        stream_handler = logging.StreamHandler(stream=stream)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(self.log_format)
        self.addHandler(stream_handler)
        return

    def set_file_handler(
        self,
        log_path:Path,
        filename:str,
        level=logging.INFO) -> None:
        # Create the path if not existing already
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # Create the file handler
        file_handler = logging.FileHandler(
            filename=log_path.with_name(filename),
            mode="a"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(self.log_format)
        self.addHandler(file_handler)
        return

def test():
    logger = SimpleLogger("TestLogger")
    logger.set_stream_handler()
    logger.info("works")

if __name__ == "__main__":
    test()
    pass
import os
import logging
from typing import Union


class Logger:
    def __init__(
        self, 
        log_file: str, 
        log_level: Union[int, str] = logging.INFO,
        is_stream_handle: bool = True,
        format: str = '%(asctime)s - %(levelname)s - %(message)s'
    ) -> None:
        if isinstance(log_level, str):
            log_level = self.set_string_level_to_int(log_level)

        os.makedirs("logs", exist_ok=True)

        # Logging properties
        log_file_path = os.path.join("logs/", log_file)
        handlers: list[logging.Handler] = [logging.FileHandler(log_file_path)]
        if is_stream_handle: handlers.append(logging.StreamHandler())

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=format,
            handlers=handlers
        )

        self.logger = logging.getLogger(__name__)    

    def set_string_level_to_int(self, level: str) -> int:
        if level == "info":
            return logging.INFO
        else:
            return logging.DEBUG

    def info(self, content: str) -> None:
        self.logger.info(content)

    def error(self, content: str) -> None:
        self.logger.error(content)


logger = Logger("log.log")
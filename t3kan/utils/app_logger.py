import logging
import logging.handlers
import os
import sys
from os.path import join
from queue import Queue
from typing import Optional

import hydra
import psutil
import torch.cuda
from colorama import Fore, Style, init

init(autoreset=True)

LOGGER_NAME = "app_logger"


class TorchFormatter(logging.Formatter):
    def format(self, record):
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024

        log_message = super().format(record)

        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024

            return f"{log_message}      | Memory: {memory_usage:.2f} MB | GPU: {gpu_memory_usage:.2f} MB / {gpu_mem_reserved:.2f} MB"

        return f"{log_message}      | Memory: {memory_usage:.2f} MB"


class TorchColoredFormatter(TorchFormatter):
    color_map = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        level_color = self.color_map.get(record.levelno, Fore.WHITE)
        record.levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"

        message_color = level_color
        record.msg = f"{message_color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


def _setup_logger(
    logger_name: str, msg_format: str, log_level: str = "INFO", file_name: Optional[str] = None
) -> logging.Logger:
    level = getattr(logging, log_level.upper().strip(), 20)

    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(level)

    if file_name:
        log_queue: Queue = Queue()
        queue_handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(queue_handler)

        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(msg_format))

        listener = logging.handlers.QueueListener(log_queue, file_handler)
        listener.start()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(TorchColoredFormatter(msg_format))
    logger.addHandler(console_handler)

    logger.debug("Logger initialized")
    return logger


@hydra.main(version_base=None, config_path="../config/model", config_name="train")
def init_logger(cfg):
    global LOGGER_NAME
    LOGGER_NAME = cfg.logging.logger_name

    if not os.path.exists(cfg.logging.log_dir):
        os.makedirs(cfg.logging.log_dir)

    _setup_logger(
        logger_name=LOGGER_NAME,
        msg_format=cfg.logging.format,
        log_level=cfg.logging.level,
        file_name=join(cfg.logging.log_dir, cfg.logging.log_file),
    )


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)

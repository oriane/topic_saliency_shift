import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Generate a unique log filename with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/app_{timestamp}.log"

    # create it if it doesn't exist
    log_path = Path(log_filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler (optional)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler]
    )


def get_logger(name):
    return logging.getLogger(name)

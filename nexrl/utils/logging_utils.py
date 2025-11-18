# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os


def set_logging_basic_config():
    """
    This function sets the global logging format and level. It will be called when import nexrl
    """
    import sys

    # Get logging level from environment variable, default to INFO
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Convert string level to logging level constant
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,  # Common alias
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,  # Common alias
    }

    level = level_mapping.get(log_level_str, logging.INFO)

    # Set the logging level while preserving the original format
    # Format: [timestamp][logger_name][level] - message
    log_stream = os.getenv("LOG_STREAM", "")

    # Explicitly set stream to stdout to ensure Ray captures the logs
    if log_stream == "stdout":
        logging.basicConfig(
            level=level,
            format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            stream=sys.stdout,
            force=True,
        )
    else:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        )

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

"""
URL utility functions for NexRL framework
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def ensure_url_scheme(url: str | None, default_scheme: str = "http") -> str:
    """
    Ensure URL has a proper scheme (http:// or https://).
    If no scheme is provided, add the default scheme.

    Args:
        url: URL string to check and potentially modify
        default_scheme: Default scheme to add if none exists (default: "http")

    Returns:
        URL with proper scheme. Returns empty string if input is None or empty.

    Examples:
        >>> ensure_url_scheme("192.168.1.1:8000")
        "http://192.168.1.1:8000"
        >>> ensure_url_scheme("http://192.168.1.1:8000")
        "http://192.168.1.1:8000"
        >>> ensure_url_scheme("https://example.com")
        "https://example.com"
        >>> ensure_url_scheme(None)
        ""
        >>> ensure_url_scheme("")
        ""
    """
    if url is None or not url:
        return ""

    url = url.strip()
    if not url:
        return ""

    # Parse the URL
    parsed = urlparse(url)

    # If no scheme is present, add the default scheme
    if not parsed.scheme:
        url = f"{default_scheme}://{url}"
    elif parsed.scheme not in ("http", "https"):
        # If scheme exists but is not http/https, warn the user
        logger.warning(
            f"URL has non-HTTP(S) scheme '{parsed.scheme}': {url}. "
            f"This may cause connection issues."
        )

    return url

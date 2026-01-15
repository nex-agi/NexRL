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

import json
import os
from typing import Any
from unittest.mock import patch

import requests  # type: ignore[import-untyped]
import urllib3


class LarkReporter:
    def __init__(self, url: str, user_name: str):
        self.url = url
        self.user_name = user_name
        USER_OPENID_FILE = os.environ.get("USER_OPENID_FILE", "")

        if not os.path.exists(USER_OPENID_FILE):
            raise ValueError(f"User openid file not found: {USER_OPENID_FILE}")

        with open(USER_OPENID_FILE, "r", encoding="utf-8") as f:
            try:
                self.user_openid = json.load(f)[self.user_name]["user_id"]
            except KeyError:
                self.user_openid = ""

    def post(self, content: str | list[list[dict]], title: str | None = None):
        """Post a message to Lark.

        When title is None, message must be a str.
        otherwise msg can be in rich text format (see
        https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/im-v1/message/create_json#45e0953e
        for details).
        """
        at_content = {"tag": "at", "user_id": self.user_openid, "user_name": "BP-OpenCompass通知"}

        msg: dict[str, Any]
        if title is None:
            assert isinstance(content, str)
            if at_content is None:
                msg = {"msg_type": "text", "content": {"text": content}}
            else:
                msg = {"msg_type": "text", "content": [[{"text": content}, at_content]]}
        else:
            if isinstance(content, str):
                if at_content is None:
                    content = [[{"tag": "text", "text": content}]]
                else:
                    content = [[{"tag": "text", "text": content}, at_content]]
            # At this point, content is guaranteed to be list[list[dict]]
            assert isinstance(content, list)
            msg = {
                "msg_type": "post",
                "content": {"post": {"zh_cn": {"title": title, "content": content}}},
            }
        try:
            with patch.dict(os.environ, {"https_proxy": os.environ.get("PROXY_URL", "")}):
                requests.post(self.url, data=json.dumps(msg), timeout=10)
        except (
            TimeoutError,
            urllib3.exceptions.MaxRetryError,
            requests.exceptions.RequestException,
        ) as e:
            print(f"Failed to connect to Lark: {str(e)}")

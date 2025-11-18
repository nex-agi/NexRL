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
Tests for config_utils module
"""

from omegaconf import OmegaConf

from nexrl.utils.config_utils import insert_config


def test_insert_config_simple():
    """Test inserting config into target"""
    target = OmegaConf.create({"a": 1, "b": 2})
    value = OmegaConf.create({"x": 10, "y": 20})

    insert_config(target, "new_key", value)

    assert target["a"] == 1
    assert target["b"] == 2
    assert target["new_key"]["x"] == 10
    assert target["new_key"]["y"] == 20


def test_insert_config_nested():
    """Test inserting nested config"""
    target = OmegaConf.create({"level1": {"a": 1}})
    value = OmegaConf.create({"nested": {"value": 100}})

    insert_config(target, "level2", value)

    assert target["level1"]["a"] == 1
    assert target["level2"]["nested"]["value"] == 100


def test_insert_config_struct_mode():
    """Test that struct mode is properly restored"""
    target = OmegaConf.create({"a": 1})
    OmegaConf.set_struct(target, True)

    value = OmegaConf.create({"x": 10})
    insert_config(target, "new_key", value, restore_struct=True)

    assert OmegaConf.is_struct(target) is True
    assert target["new_key"]["x"] == 10

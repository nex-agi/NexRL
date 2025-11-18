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
Unit tests for k8s_utils module
"""

import unittest
from unittest.mock import MagicMock, patch

from cli.utils import k8s_utils


class TestK8sUtils(unittest.TestCase):
    """Test cases for K8s utilities"""

    @patch("subprocess.run")
    def test_check_kubectl_available_success(self, mock_run):
        """Test kubectl availability check - success case"""
        mock_run.return_value = MagicMock(returncode=0)
        result = k8s_utils.check_kubectl_available()
        self.assertTrue(result)
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_check_kubectl_available_failure(self, mock_run):
        """Test kubectl availability check - failure case"""
        mock_run.return_value = MagicMock(returncode=1)
        result = k8s_utils.check_kubectl_available()
        self.assertFalse(result)

    @patch("subprocess.run")
    def test_check_namespace_exists(self, mock_run):
        """Test namespace existence check"""
        mock_run.return_value = MagicMock(returncode=0)
        result = k8s_utils.check_namespace_exists("test-namespace")
        self.assertTrue(result)
        mock_run.assert_called_with(
            ["kubectl", "get", "namespace", "test-namespace"],
            capture_output=True,
            text=True,
            timeout=10,
        )

    @patch("subprocess.run")
    def test_create_namespace(self, mock_run):
        """Test namespace creation"""
        mock_run.return_value = MagicMock(returncode=0)
        result = k8s_utils.create_namespace("test-namespace")
        self.assertTrue(result)

    @patch("subprocess.run")
    def test_apply_yaml_success(self, mock_run):
        """Test YAML application - success case"""
        mock_run.return_value = MagicMock(returncode=0, stdout="created")
        yaml_content = "apiVersion: v1\nkind: ConfigMap"
        result = k8s_utils.apply_yaml(yaml_content, "test-namespace")
        self.assertTrue(result)


class TestK8sResourceManagement(unittest.TestCase):
    """Test cases for K8s resource management"""

    @patch("subprocess.run")
    def test_get_configmap(self, mock_run):
        """Test ConfigMap retrieval"""
        mock_json = '{"data": {"key": "value"}}'
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_json)

        result = k8s_utils.get_configmap("test-config", "test-namespace")
        self.assertIsNotNone(result)
        self.assertIn("data", result)
        self.assertEqual(result["data"]["key"], "value")

    @patch("subprocess.run")
    def test_delete_resource(self, mock_run):
        """Test resource deletion"""
        mock_run.return_value = MagicMock(returncode=0)
        result = k8s_utils.delete_resource("deployment", "test-app", "test-namespace")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()

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
Unit tests for validation module
"""

import unittest
from unittest.mock import MagicMock, patch

from cli.utils import validation


class TestValidation(unittest.TestCase):
    """Test cases for validation utilities"""

    @patch("cli.utils.k8s_utils.check_kubectl_available")
    @patch("cli.utils.k8s_utils.check_namespace_exists")
    @patch("cli.utils.k8s_utils.get_configmap")
    @patch("cli.utils.k8s_utils.get_secret")
    def test_validate_admin_setup_success(self, mock_secret, mock_cm, mock_ns, mock_kubectl):
        """Test admin setup validation - success case"""
        mock_kubectl.return_value = True
        mock_ns.return_value = True
        mock_cm.return_value = {"data": {}}
        mock_secret.return_value = {"data": {}}

        is_valid, message = validation.validate_admin_setup("test-namespace")
        self.assertTrue(is_valid)

    @patch("cli.utils.k8s_utils.check_kubectl_available")
    def test_validate_admin_setup_no_kubectl(self, mock_kubectl):
        """Test admin setup validation - kubectl not available"""
        mock_kubectl.return_value = False

        is_valid, message = validation.validate_admin_setup("test-namespace")
        self.assertFalse(is_valid)
        self.assertIn("kubectl", message)

    def test_validate_storage_path_absolute(self):
        """Test storage path validation - absolute path"""
        is_valid, message = validation.validate_storage_path("/gpfs/data/nexrl")
        self.assertTrue(is_valid)

    def test_validate_storage_path_relative(self):
        """Test storage path validation - relative path"""
        is_valid, message = validation.validate_storage_path("relative/path")
        self.assertFalse(is_valid)
        self.assertIn("absolute", message)

    def test_validate_storage_path_empty(self):
        """Test storage path validation - empty path"""
        is_valid, message = validation.validate_storage_path("")
        self.assertFalse(is_valid)
        self.assertIn("empty", message)

    @patch("cli.utils.k8s_utils.check_crd_exists")
    def test_check_volcanojob_support(self, mock_crd):
        """Test VolcanoJob CRD check"""
        mock_crd.return_value = True
        is_valid, message = validation.check_volcanojob_support()
        self.assertTrue(is_valid)
        self.assertIn("available", message)


class TestDockerImageValidation(unittest.TestCase):
    """Test cases for Docker image validation"""

    def test_validate_docker_images_complete(self):
        """Test Docker image validation - all images present"""
        images = {
            "train_router_image": "registry/train-router:latest",
            "worker_image": "registry/worker:latest",
            "controller_image": "registry/controller:latest",
            "inference_image": "registry/inference:latest",
        }
        is_valid, issues = validation.validate_docker_images(images)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

    def test_validate_docker_images_missing(self):
        """Test Docker image validation - missing image"""
        images = {
            "train_router_image": "registry/train-router:latest",
            "worker_image": "registry/worker:latest",
        }
        is_valid, issues = validation.validate_docker_images(images)
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)


if __name__ == "__main__":
    unittest.main()

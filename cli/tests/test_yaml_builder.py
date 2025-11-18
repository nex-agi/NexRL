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
Unit tests for yaml_builder module
"""

import unittest

from cli.utils import yaml_builder


class TestYAMLBuilder(unittest.TestCase):
    """Test cases for YAML template rendering"""

    def test_render_redis_secret(self):
        """Test Redis secret YAML rendering"""
        redis_config = {
            "host": "redis.example.com",
            "port": "6379",
            "username": "user",
            "password": "pass",
        }

        yaml_content = yaml_builder.render_redis_secret("test-ns", redis_config)

        self.assertIn("nexrl-redis-secret", yaml_content)
        self.assertIn("test-ns", yaml_content)
        self.assertIn("redis.example.com", yaml_content)
        self.assertIn("6379", yaml_content)

    def test_render_train_router_deployment(self):
        """Test train router deployment YAML rendering"""
        yaml_content = yaml_builder.render_train_router_deployment(
            namespace="test-ns", image="test-image:latest"
        )

        self.assertIn("train-router", yaml_content)
        self.assertIn("test-ns", yaml_content)
        self.assertIn("test-image:latest", yaml_content)
        self.assertIn("Deployment", yaml_content)

    def test_render_train_router_service(self):
        """Test train router service YAML rendering"""
        yaml_content = yaml_builder.render_train_router_service("test-ns")

        self.assertIn("train-router-svc", yaml_content)
        self.assertIn("test-ns", yaml_content)
        self.assertIn("Service", yaml_content)
        self.assertIn("8000", yaml_content)

    def test_render_routers_configmap(self):
        """Test routers ConfigMap YAML rendering"""
        yaml_content = yaml_builder.render_routers_configmap("test-ns")

        self.assertIn("nexrl-routers-config", yaml_content)
        self.assertIn("test-ns", yaml_content)
        self.assertIn("train_router_url", yaml_content)
        self.assertIn("rollout_router_url", yaml_content)

    def test_render_worker_volcanojob(self):
        """Test worker VolcanoJob YAML rendering"""
        yaml_content = yaml_builder.render_worker_volcanojob(
            job_name="test-job",
            namespace="test-ns",
            identifier="test-id",
            worker_image="worker:latest",
            train_router_url="http://train-router:8000",
            worker_config_path="/app/config.yaml",
            storage_path="/gpfs/data",
            world_size=4,
            gpus_per_pod=8,
        )

        self.assertIn("test-job-workers", yaml_content)
        self.assertIn("VolcanoJob", yaml_content)
        self.assertIn("test-ns", yaml_content)
        self.assertIn("worker:latest", yaml_content)
        self.assertIn("test-id", yaml_content)

    def test_render_nexrl_controller_volcanojob(self):
        """Test NexRL controller VolcanoJob YAML rendering"""
        yaml_content = yaml_builder.render_nexrl_controller_volcanojob(
            job_name="test-job",
            namespace="test-ns",
            identifier="test-id",
            controller_image="controller:latest",
            train_router_url="http://train-router:8000",
            rollout_router_url="http://rollout-router:38848",
            storage_path="/gpfs/data",
            job_config_path="/app/config",
            PROJECT_NAME="TestProject",
            EXPERIMENT_NAME="test-exp",
            TRAIN_STEPS=100,
        )

        self.assertIn("test-job-controller", yaml_content)
        self.assertIn("VolcanoJob", yaml_content)
        self.assertIn("controller:latest", yaml_content)
        self.assertIn("TestProject", yaml_content)


if __name__ == "__main__":
    unittest.main()

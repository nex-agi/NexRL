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
Simple test for TinkerInferenceServiceClient

Tests that generate() and completion() APIs return values normally.
Similar to .vibe/tinker/test_sampling.py
"""

import os

# Set your API key here (replace with your actual key)
API_KEY = ""
os.environ["TINKER_API_KEY"] = API_KEY

from omegaconf import OmegaConf

from nexrl.tinker import TinkerInferenceServiceClient, TinkerServiceHolder
from nexrl.utils.logging_utils import set_logging_basic_config


def test_inference_client():
    set_logging_basic_config()

    print("🔧 TinkerInferenceServiceClient Test\n")

    # Configuration
    model_name = "Qwen/Qwen3-8B"
    tokenizer_path = "Qwen/Qwen3-8B"

    # Create TinkerServiceHolder (now owns tokenizer and renderer)
    print("1. Creating TinkerServiceHolder...")
    service_holder = TinkerServiceHolder(
        base_model=model_name,
        lora_rank=32,
        tokenizer_path=tokenizer_path,
    )
    print("   ✓ TinkerServiceHolder created\n")

    # Create config for inference client (no tokenizer needed, it's in the holder)
    config = OmegaConf.create(
        {
            "inference_service": {
                "model": model_name,
                "model_tag": "default",
                "max_tokens": 100,
                "freeze_for_weight_sync": False,  # Disable for simple testing
            },
            "temperature": 0.0,  # Greedy for deterministic output
        }
    )

    # Create TinkerInferenceServiceClient
    print("2. Creating TinkerInferenceServiceClient...")
    client = TinkerInferenceServiceClient(config, service_holder)
    print("   ✓ TinkerInferenceServiceClient created\n")

    # Test generate() - chat completion
    print("3. Testing generate() API...")
    messages = [{"role": "user", "content": "What is 2 + 2? Answer briefly."}]
    print(f"   Input: {messages}")

    generate_result = client.generate(messages)

    print("\n   === generate() Output ===")
    print(f"   Keys: {list(generate_result.keys())}")
    print("\n   Values:")
    for key, value in generate_result.items():
        print(f"   [{key}]: {value}")
    print("\n   ✓ generate() returned successfully\n")

    # Test completion() - prompt completion
    print("4. Testing completion() API...")
    prompt = "The capital of France is"
    print(f"   Input: {prompt}")

    completion_result = client.completion(prompt)

    print("\n   === completion() Output ===")
    print(f"   Keys: {list(completion_result.keys())}")
    print("\n   Values:")
    for key, value in completion_result.items():
        print(f"   [{key}]: {value}")
    print("\n   ✓ completion() returned successfully\n")

    # Summary
    print("=" * 50)
    print("🎉 All tests passed!")
    print("=" * 50)

    return generate_result, completion_result


if __name__ == "__main__":
    test_inference_client()

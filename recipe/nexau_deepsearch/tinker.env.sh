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

set -e

echo "=========================================="
echo "Setting up environment for DeepSearch (Tinker)"
echo "=========================================="


# Tinker API Key (REQUIRED for Tinker mode)
export TINKER_API_KEY="tml-your-tinker-api-key-here"

# Web Search API Key (REQUIRED for DeepSearch)
export SERPER_API_KEY="your-serper-api-key"

# LangFuse Monitoring (Optional)
export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"


echo "=========================================="
echo "✓ DeepSearch (Tinker) environment setup complete"
echo "=========================================="

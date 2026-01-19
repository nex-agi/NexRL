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
echo "Setting up environment for DeepSearch (Weaver)"
echo "=========================================="

# Weaver API Key (REQUIRED for Weaver mode)
export WEAVER_API_KEY="sk-your-weaver-api-key-here"

# Web Search API Key (REQUIRED for DeepSearch)
export SERPER_API_KEY="your-serper-api-key"
export LLM_MODEL=${SERVED_MODEL_NAME}
export LLM_API_KEY="EMPTY"
export LLM_BASE_URL="EMPTY"


# LangFuse Monitoring (Optional)
export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"


echo "=========================================="
echo "✓ DeepSearch (Weaver) environment setup complete"
echo "=========================================="

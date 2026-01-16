# Agent Configuration

This document explains how to configure NexAU agents for your NexRL tasks.

## Overview

NexAU agents are configured via YAML files in the `agent_workspace/` directory. The agent configuration defines the system prompt, LLM settings, tools, and tracing infrastructure.

## Configuration File Location

Agent configuration is typically placed at:

```
recipe/my_task/agent_workspace/agent_config.yaml
```

Referenced from recipe configuration:

```yaml
rollout_worker:
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
```

## Basic Configuration Structure

```yaml
type: agent
name: my_agent_name
max_context_tokens: 100000
system_prompt: "Your system prompt..."
system_prompt_type: string
tool_call_mode: openai

llm_config:
  temperature: 0.7
  max_tokens: 8192
  api_type: openai_chat_completion

tracers:
  - import: nexau.archs.tracer.adapters.in_memory:InMemoryTracer
```

## Configuration Fields

### Agent Identity

**`type`** (required)
- Value: `"agent"`
- Identifies this as an agent configuration

**`name`** (required)
- Agent name for identification and logging
- Example: `"news_recommendation_agent"`

### Context Management

**`max_context_tokens`** (required)
- Maximum context window size in tokens
- Common values: `100000`, `200000`
- Should match or be less than the LLM's context limit

### System Prompt

**`system_prompt`** (required)
- The system instruction that defines agent behavior
- Can be multi-line string with task-specific instructions
- Use clear, structured prompts for best results

**`system_prompt_type`** (required)
- Value: `"string"` (direct string prompt)
- Indicates how to interpret the system prompt field

### LLM Configuration

**`llm_config`** (required)
- Configures the language model settings

**`llm_config.temperature`**
- Sampling temperature (0.0 to 2.0)
- Lower = more deterministic, Higher = more creative
- Typical: `0.7` for most tasks

**`llm_config.max_tokens`**
- Maximum tokens in LLM response
- Common: `8192`, `4096`
- Should fit within your inference service limits

**`llm_config.api_type`**
- Value: `"openai_chat_completion"`
- Specifies OpenAI-compatible chat completion API

### Tool Configuration

**`tool_call_mode`** (optional)
- Value: `"openai"`
- Enables OpenAI function calling format for tools

**`tools`** (optional)
- List of tool configurations
- Each tool references a YAML definition file

### Tracing

**`tracers`** (required)
- List of tracer configurations for monitoring agent execution
- Common: `InMemoryTracer` for trajectory capture

## Real Examples

### News Recommendation Agent

From `recipe/nexau_news/agent_workspace/agent_config.yaml`:

```yaml
type: agent
name: news_recommendation_agent
max_context_tokens: 100000
system_prompt: |
  你是园区招商经理的新闻审批助理，招商经理最终要达成的目的有：
  第一，只推送重要新闻。
  第二，之前已经推送的这家公司同一主题新闻，不需要再推送。
  ...
  思考再回答，把你的思考过程放到<think></think>格式里。
  把你的答案放到<answer></answer>里。<answer>里只能有True或False。
system_prompt_type: string
tool_call_mode: openai

llm_config:
  temperature: 0.7
  max_tokens: 8192
  api_type: openai_chat_completion

tracers:
  - import: nexau.archs.tracer.adapters.in_memory:InMemoryTracer
```

**Key Features:**
- Chinese language system prompt with structured output format
- Requires reasoning in `<think>` tags and answer in `<answer>` tags
- Uses InMemoryTracer to capture agent trajectories for training

### Deep Search Agent

From `recipe/nexau_deepsearch/agent_workspace/agent_config.yaml`:

```yaml
type: agent
name: deepsearch_agent
max_context_tokens: 100000
system_prompt: |
  You are a helpful research assistant. Your task is to answer questions
  by searching the web and synthesizing information.

  Put your final answer in <answer></answer> tags.
system_prompt_type: string
tool_call_mode: openai

llm_config:
  temperature: 0.7
  max_tokens: 8192
  api_type: openai_chat_completion

tools:
  - tools/WebSearch.yaml

tracers:
  - import: nexau.archs.tracer.adapters.in_memory:InMemoryTracer
```

**Key Features:**
- Includes custom web search tool
- Structured output format for answer extraction
- Tool calling enabled via OpenAI mode

## System Prompt Design

### Structured Output Format

Use XML tags to structure agent responses for easier parsing:

```yaml
system_prompt: |
  Your task instructions here.

  Format your response as:
  <think>Your reasoning process</think>
  <answer>Your final answer</answer>
```

Benefits:
- Easy to extract answers with regex
- Encourages chain-of-thought reasoning
- Consistent format across tasks

### Task-Specific Instructions

Include clear guidelines for your specific task:

```yaml
system_prompt: |
  You are a math problem solver.

  Instructions:
  1. Read the problem carefully
  2. Show your work step by step
  3. Provide the final numerical answer

  Format:
  <reasoning>Show your steps here</reasoning>
  <answer>Final numerical answer only</answer>
```

### Multi-Language Support

NexAU supports prompts in any language:

```yaml
system_prompt: "你是一个数学助手。请解决以下问题..."  # Chinese
# or
system_prompt: "You are a math assistant. Solve the following..."  # English
```

## Tool Configuration

### Defining Tools

Tools are defined in separate YAML files under `agent_workspace/tools/`:

```
agent_workspace/
├── agent_config.yaml
└── tools/
    └── WebSearch.yaml
```

Reference tools in agent config:

```yaml
tools:
  - tools/WebSearch.yaml
```

### Tool Definition Format

Example `tools/WebSearch.yaml`:

```yaml
type: function
function:
  name: web_search
  description: Search the web for information
  parameters:
    type: object
    properties:
      query:
        type: string
        description: Search query
    required:
      - query
```

### Custom Tool Implementation

Implement tool logic in Python:

```python
# agent_workspace/web_tool.py
def web_search(query: str) -> str:
    # Your implementation
    results = search_api.search(query)
    return format_results(results)
```

Register in agent workspace and ensure NexAU agent has access.

## Tracer Configuration

### InMemoryTracer

Default tracer for capturing agent trajectories:

```yaml
tracers:
  - import: nexau.archs.tracer.adapters.in_memory:InMemoryTracer
```

Captures:
- Agent actions
- Tool calls
- LLM responses
- Final answers

Used by rollout workers to extract trajectories for training.

### LangFuse Tracer (Optional)

For external monitoring and debugging:

```yaml
tracers:
  - import: nexau.archs.tracer.adapters.langfuse:LangfuseTracer
    params:
      public_key: ${env.LANGFUSE_PUBLIC_KEY}
      secret_key: ${env.LANGFUSE_SECRET_KEY}
      host: ${env.LANGFUSE_BASE_URL}
  - import: nexau.archs.tracer.adapters.in_memory:InMemoryTracer
```

**Note:** Always include `InMemoryTracer` for trajectory capture, even if using additional tracers.

## Environment Variable Interpolation

Agent configs support environment variable substitution:

```yaml
tracers:
  - import: nexau.archs.tracer.adapters.langfuse:LangfuseTracer
    params:
      public_key: ${env.LANGFUSE_PUBLIC_KEY}
      secret_key: ${env.LANGFUSE_SECRET_KEY}
      host: ${env.LANGFUSE_BASE_URL}
```

Set in environment setup script:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

## Integration with Rollout Workers

### Configuration Reference

In recipe `common.yaml`:

```yaml
rollout_worker:
  type: "nexau"
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
  evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"
  task_name: "my_task"
```

### Agent Loading

The `BaseNexAURolloutWorker`:
1. Loads agent config from specified path
2. Initializes NexAU agent with configuration
3. Uses InMemoryTracer to capture trajectories
4. Passes trajectories to evaluator

### Trajectory Extraction

Agent traces are processed into trajectories:

```python
# In BaseNexAURolloutWorker
def _process_observation(self, observation):
    # Extract from InMemoryTracer
    final_answer = self._extract_final_answer(observation)

    # Create evaluation target
    evaluation_target = NexAUEvaluationTarget(
        final_answer=final_answer,
        observation=observation
    )

    # Pass to evaluator
    result = self.evaluator.evaluate(data, evaluation_target)
```

## Best Practices

### System Prompt Design

1. **Be specific** - Clear task description and requirements
2. **Use structure** - XML tags for consistent output format
3. **Include examples** - Few-shot examples in prompt when helpful
4. **Set constraints** - Length limits, format requirements
5. **Encourage reasoning** - Request chain-of-thought process

### LLM Settings

1. **Temperature**
   - Use 0.7-1.0 for creative tasks
   - Use 0.3-0.5 for deterministic tasks
   - Use 0.0 for maximum consistency

2. **Max Tokens**
   - Set based on expected response length
   - Leave buffer for tool calls and reasoning
   - Common: 8192 for most tasks

3. **Context Window**
   - Use full context for complex multi-turn tasks
   - Smaller for simple single-turn tasks
   - Monitor token usage in logs

### Tools

1. **Only include necessary tools** - Reduces complexity
2. **Clear descriptions** - Help agent choose correct tool
3. **Validate tool outputs** - Handle errors gracefully
4. **Test tools independently** - Before integration

### Tracing

1. **Always use InMemoryTracer** - Required for training
2. **Add external tracers for debugging** - LangFuse, etc.
3. **Monitor trace quality** - Ensure complete captures
4. **Clean up traces** - Prevent memory leaks in long runs

## Troubleshooting

### Agent Not Loading

**Problem:** Agent configuration file not found

**Solution:**
- Check path in recipe config is correct
- Ensure path is relative to recipe directory
- Verify file exists at expected location

### Tracer Not Capturing

**Problem:** No trajectories produced

**Solution:**
- Verify InMemoryTracer is included
- Check agent completes execution
- Ensure no exceptions during agent run
- Review logs for tracer errors

### Tool Calls Failing

**Problem:** Agent can't use tools

**Solution:**
- Set `tool_call_mode: openai` in agent config
- Verify tool definition format
- Check tool implementation is accessible
- Test tool independently

### Output Format Issues

**Problem:** Evaluator can't extract answers

**Solution:**
- Review system prompt output format instructions
- Check regex patterns in evaluator match prompt format
- Test prompt with manual examples
- Add format validation in evaluator

## Related Documentation

- [Recipe Structure](./recipe-structure.md) - Overall recipe organization
- [Recipe Configuration](./recipe-configuration.md) - YAML configuration details
- [NexAU Rollout Worker](../05-rollout-workers/nexau-rollout-worker.md) - Agent integration
- [Evaluators](../05-rollout-workers/evaluators.md) - Evaluation implementation

---

**Next**: [Recipe Configuration](./recipe-configuration.md) - Learn about YAML configuration and Hydra composition.

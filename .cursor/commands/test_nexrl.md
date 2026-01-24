---
name: Test-NexRL
description: How to test NexRL end-to-end in self-hosted mode
---

## Background(You MUST be careful)
NexRL is an RL framework, so when running end-to-end test, it involves multiple Nodes, GPUs, CPUs distributed across the entire cluster.

**You Must FOLLOW the rules to run the test**
assuming run tests with config `recipe/on_policy_distill/opd_qwen2a5_3b_from_14b.yaml`
<1>. setting up and ask for resources(GPUs, CPUs)
```bash
python scripts/run.py --mode self-hosted --train-config recipe/on_policy_distill/opd_qwen2a5_3b_from_14b.yaml
```
<2>. wait until step <1> is finished(otherwise you won't have the resource to launch tests)
from the terminal info of <1>, it will states:
"""
To manually run the experiment:
  kubectl exec -it ${job_id} -n qiji -- bash
Then inside the pod:
  bash scripts/common/run_nexrl.sh
"""
so you follow the instructions
```bash
kubectl exec -it ${job_id} -n qiji -- bash
```
after you enter the pod, then
```bash
bash scripts/common/run_nexrl.sh
```

now the experiments starts.

<3>. how to check the status of the tests/experiments
since we run with config
`recipe/on_policy_distill/opd_qwen2a5_3b_from_14b.yaml`, the logs will be under `logs/opd-qwen2a5-3b-from-14b/`
**Important 1:** maybe we always test the same config `opd_qwen2a5_3b_from_14b.yaml`, so all the tests log will be under `logs/opd-qwen2a5-3b-from-14b/`, **always** check the **last** sub folder under `logs/opd-qwen2a5-3b-from-14b/`, your latest test logs is here!

**Important 2:** while checking logs for error information, you may need to perform some search/grep(find errors), otherwise the `context`(texts) might be extremely long.

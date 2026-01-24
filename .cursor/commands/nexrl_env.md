---
name: NexRL-Env
description: When you need to run some commands(`python` or `bash`), you must start a termianl and activate env.
---

## How to run commands(`python` or `bash`)
1. Everytime you need to run some commands (`python` or `bash`), expecially they are related to `NexRL`, you MUST do:
- start a terminal
- navigate to `NexRL` folder: /gpfs/users/jiangzhen/jzdev/NexRL/
- activate conda env
```bash
source /gpfs/users/jiangzhen/miniconda3/bin/activate && conda activate rl
```
**when start a terminal(you are on CPU-Only dev pod), so you can't run GPU-related commands. e.g. `PyTorch` related projects that using CUDA/GPUs**

---
name: Using-Proxy
description: Running scripts or commands when proxy are required. Usually happens when `install` or `download`, without proxy, errors or timeout will occur.
---

## How to enable proxy
1. You MUST know that the user develop in a `dev` pod, only has CPUs.
2. Every time you start a new terminal, if your commands/scripts requires networks, you may encounter:
    - network/remote error
    - network/remote timeout
    - high network latency
    - extremely low downloading speed
if those happens, you need to consider using the following proxy.
3. You are provided with two proxies, sometime they both work, sometime only one of them will work, you need to try them.
**If all of them fails, you MUST report.**

```shell
# proxy 1
export http_proxy=http://10.1.2.1:7890 && export https_proxy=http://10.1.2.1:7890
# proxy 2
export http_proxy=http://10.51.6.23:1091 && export https_proxy=http://10.51.6.23:1091
```

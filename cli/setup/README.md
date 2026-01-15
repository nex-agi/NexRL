# Production Setup

> **Note:** For testing, skip this! Just run `nexrl -m self-hosted -c config.yaml --run-nexrl`

---

## When You Need This

✅ Multi-user cluster
✅ Custom Docker images
✅ Persistent shared storage
✅ Team deployment
✅ Shared inference router (optional)

Otherwise use defaults or environment variables ([../README.md](../README.md))

---

## Setup Options

### Option A: Basic Setup (5 Steps)

For single-user or testing environments.

### Option B: Full Setup with Router (8 Steps)

For multi-user production with shared inference routing.

---

## Basic Setup (6 Steps)

### 1. Namespace
```bash
kubectl apply -f 01-namespace.yaml
```

### 2. Job RBAC (Required)
**This creates a service account with permissions to create Volcano jobs:**
```bash
kubectl apply -f 09-nexrl-job-rbac.yaml
```

### 3. Admin Config
**Edit `02-admin-config.yaml`:**
```yaml
storage_path: "/your/nfs/path"     # ← Change!
worker_image: "your-reg.com/img"    # ← Change!
controller_image: "your-reg.com/img" # ← Change!
```

**Apply:**
```bash
kubectl apply -f 02-admin-config.yaml
```

### 4. User Secret
```bash
cp 03-user-config-template.yaml 03-user-config.yaml
vim 03-user-config.yaml  # Change YOUR_USER_ID and YOUR_WANDB_KEY
kubectl apply -f 03-user-config.yaml
```

### 5. Local Config (Optional)
```bash
mkdir -p ~/.nexrl
echo "user_id: alice" > ~/.nexrl/config.yaml
```

### 6. Volcano Queue
```bash
kubectl apply -f 04-volcano-queue.yaml
```

---

## Full Setup with Router (9 Steps)

If you want a shared inference router for all jobs:

### 7. Redis Secret (for Router)
**Edit `05-redis-secret-template.yaml`:**
```yaml
stringData:
  REDIS_HOST: "your-redis-host"    # ← Change!
  REDIS_PORT: "6379"               # ← Change if needed
  REDIS_PASSWORD: "your-password"  # ← Change!
```

**Apply:**
```bash
kubectl apply -f 05-redis-secret-template.yaml
```

### 8. Router RBAC
```bash
kubectl apply -f 06-sglang-router-rbac.yaml
```

### 9. Rollout Router & ConfigMap
```bash
kubectl apply -f 07-rollout-router.yaml
kubectl apply -f 08-routers-configmap.yaml
```

**What does this do?**
- Creates a shared SGLang router that load-balances across all inference instances
- Router performs service discovery and health checking
- All jobs share the same router URL: `http://rollout-router-svc.nexrl.svc.cluster.local:12345`

**When to use?**
- ✅ Multi-user environments with shared inference
- ✅ Production deployments requiring load balancing
- ✅ When you have Redis available

**When to skip?**
- ❌ Single-user testing
- ❌ No Redis available
- ❌ Prefer per-job isolated routing

If you skip router setup, NexRL will automatically create a dynamic router per job.

---

## Verify

### Basic Setup
```bash
kubectl get namespace nexrl
kubectl get configmap nexrl-admin-settings -n nexrl
kubectl get secret nexrl-wandb-alice -n nexrl
```

### Full Setup (with Router)
```bash
kubectl get namespace nexrl
kubectl get configmap nexrl-admin-settings -n nexrl
kubectl get secret nexrl-wandb-alice -n nexrl
kubectl get secret nexrl-redis-secret -n nexrl
kubectl get deployment rollout-router -n nexrl
kubectl get service rollout-router-svc -n nexrl
```

---

## Router Behavior

### With Pre-deployed Router (kubectl apply -f cli/setup/)
- All jobs share the same router at `rollout-router-svc`
- Router performs service discovery and load balancing
- `INFERENCE_BASE_URL` = `http://rollout-router-svc.nexrl.svc.cluster.local:12345`

### Without Pre-deployed Router (default)
- Each job gets its own dynamic router
- Router lifecycle tied to the job
- `INFERENCE_BASE_URL` = `http://<router-pod-ip>:12345`

---

## Multi-User

**Admin (once - Basic):**
```bash
kubectl apply -f 01-namespace.yaml 09-nexrl-job-rbac.yaml 02-admin-config.yaml 04-volcano-queue.yaml
```

**Admin (once - With Router):**
```bash
kubectl apply -f 01-namespace.yaml 09-nexrl-job-rbac.yaml 02-admin-config.yaml 04-volcano-queue.yaml \
               05-redis-secret-template.yaml 06-sglang-router-rbac.yaml \
               07-rollout-router.yaml 08-routers-configmap.yaml
```

**Each user:**
```bash
cp 03-user-config-template.yaml user-alice.yaml
vim user-alice.yaml
kubectl apply -f user-alice.yaml
```

---

## Update

```bash
# Update images
kubectl edit configmap nexrl-admin-settings -n nexrl

# Update user secret
kubectl delete secret nexrl-wandb-alice -n nexrl
kubectl apply -f 03-user-config.yaml
```

---

## Advanced

### Private Registry
```bash
kubectl create secret docker-registry regcred \
  --docker-server=reg.com --docker-username=u --docker-password=p -n nexrl
```

### NFS Storage
```yaml
storage_type: "nfs"
nfs_server: "server.com"
nfs_path: "/exports/nexrl"
```

### PVC Storage
```yaml
storage_type: "pvc"
storage_pvc_name: "nexrl-storage"
```

---

## Troubleshooting

```bash
# Check ConfigMap
kubectl get configmap -n nexrl

# Check Secret
kubectl get secret -n nexrl | grep wandb

# Test storage
kubectl run test --rm -it --image=busybox -n nexrl -- ls /storage
```

---

## Cleanup

```bash
kubectl delete namespace nexrl
```

---

**Main CLI Docs:** [../README.md](../README.md)

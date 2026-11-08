# hunkon_rag · Kubernetes 部署手册

> **诚实标注**：本手册对应的 K8s 清单（`k8s/` 目录）已按生产规范编写（Deployment / Service / HPA / Ingress / Secret），
> **但未在真实集群中执行**——编写环境（Windows 本机）无 Docker 守护进程，无法启动 kind/minikube 本地集群。
> 清单已通过 `kubectl apply --dry-run=client` 级语法校验（若可用），并遵循官方 API 规范。
> 在有 Docker 或云 K8s 的环境中，按本手册即可一键部署。

---

## 一、架构图

```
                    ┌──────────────────────────────────────────┐
                    │           Kubernetes 集群                 │
                    │                                          │
  Ingress ──▶ Service ──▶ Deployment (2+ 副本)                 │
  (rag.example.com)  │         │                              │
                    │         ├── Pod: hunkon-rag (FastAPI:8000) │
                    │         ├── Pod: hunkon-rag               │
                    │         └── ...（HPA 自动扩到最多 6）     │
                    │                                          │
                    │  HorizontalPodAutoscaler                 │
                    │  (CPU > 70% → 扩容)                      │
                    │                                          │
                    │  Secret (API Key / LLM Key)              │
                    └──────────────────────────────────────────┘
```

| 组件 | 作用 |
|---|---|
| **Deployment** | 声明副本数（2）、资源限制、健康检查（readiness/liveness） |
| **Service** | 集群内稳定访问入口（ClusterIP，端口 80 → 容器 8000） |
| **HPA** | CPU 超过 70% 自动扩容，最大 6 副本 |
| **Ingress** | 对外 HTTP 暴露（需 Ingress Controller，如 ingress-nginx） |
| **Secret** | 敏感配置（RAG_API_KEY、RAG_LLM_API_KEY）从 Secret 注入，不写进镜像 |

---

## 二、前置条件

- Kubernetes 集群（云厂商 EKS/ACK/GKE 或本地 kind/minikube）
- `kubectl` 已配置连接集群
- metrics-server 已部署（HPA 依赖，`kubectl top nodes` 可用）
- Ingress Controller（仅当需要对外暴露）

---

## 三、构建镜像

```bash
# 在 hunkon_rag 仓库根目录
docker build -t hunkon-rag:latest .
```

> 若推送到远程仓库（如 Docker Hub / 云厂商 ACR），改为 `docker build -t <你的镜像仓库>/hunkon-rag:latest .` 并 push，
> 然后修改 `k8s/deployment.yaml` 的 `image` 字段。

---

## 四、配置 Secret

```bash
cd k8s
cp secret.example.yaml secret.yaml
# 编辑 secret.yaml，填入 base64 编码的真实 Key：
echo -n '你的APIKey' | base64
echo -n '你的LLMKey' | base64
```

---

## 五、部署

```bash
# 方式一：kustomize 一键应用
kubectl apply -k k8s/

# 方式二：逐个应用
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml
```

---

## 六、验证

```bash
# 查看 Pod 状态（等待 Running + Ready 1/1）
kubectl get pods -l app=hunkon-rag

# 查看 Service
kubectl get svc hunkon-rag

# 查看 HPA 状态
kubectl get hpa hunkon-rag

# 集群内访问（进入任意 Pod curl 服务）
kubectl exec -it <pod名> -- curl http://hunkon-rag/health

# 查看日志
kubectl logs -l app=hunkon-rag

# 查看 HPA 是否生效（CPU 使用率）
kubectl top pods -l app=hunkon-rag
```

预期 `/health` 返回：
```json
{"status":"ok","llm":"mock","embedder":"mock","model":"gpt-4o-mini"}
```

---

## 七、水平扩缩容验证

```bash
# 制造 CPU 压力（可选：用 hey/locust 压测 /chat 接口）
# 观察 HPA 是否自动扩容：
kubectl get hpa hunkon-rag -w

# 预期：CPU 超过 70% 后，副本从 2 逐步扩到 3/4/.../6
# 压力下降后，缩容回 minReplicas
```

---

## 八、常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| Pod 一直 Pending | 资源不足 / 无可用节点 | `kubectl describe pod` 查看事件，检查节点资源 |
| Pod CrashLoopBackOff | 启动失败（如 Secret 未配置） | `kubectl logs` 查看，确认 Secret 已注入 |
| HPA 不工作 | metrics-server 未部署 | 部署 metrics-server，`kubectl top nodes` 验证 |
| Ingress 不生效 | 无 Ingress Controller / DNS 未解析 | 部署 ingress-nginx，配置 DNS 指向集群入口 |
| 503 Service Unavailable | 就绪探针未通过，Pod 未就绪 | `kubectl describe pod` 查看 readiness 状态 |

---

## 九、生产环境补充（诚实边界）

本清单是**可部署的基线**，生产环境还应补充：

- **TLS**：Ingress 配置 HTTPS 证书（cert-manager + Let's Encrypt）
- **持久化**：SQLite 不适合生产多副本（应换 PostgreSQL/pgvector），并配置 PVC
- **配置管理**：非敏感配置建议用 ConfigMap（当前直接写在 deployment env）
- **CI/CD**：GitHub Actions 构建镜像 + 推送到仓库 + `kubectl rollout restart` 自动部署
- **可观测**：接入 Prometheus 采集 /metrics，Grafana 看板，告警规则

---

## 十、本仓库落地状态

| 项 | 状态 |
|---|---|
| Dockerfile | ✅ 已有（python:3.13-slim，端口 8000） |
| k8s/ 清单 | ✅ 已编写（Deployment/Service/HPA/Ingress/Secret/Kustomize） |
| 真实集群验证 | ⚠️ 未执行（本机无 Docker 守护进程） |
| 镜像推送 | ⚠️ 未推送（无远程仓库） |

> 在有 Docker 或云 K8s 的环境下，按本手册可完成「构建镜像 → 配置 Secret → kubectl apply → 验证 → HPA 扩缩容」全流程。
> 本手册记录 K8s 部署设计，但**不声明已在生产集群运行**。
# 部署

## Docker

```bash
docker build -f deploy/docker/Dockerfile -t terraforge .
docker run -p 8000:8000 terraforge
```

## Kubernetes

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

## Helm

```bash
helm install terraforge deploy/helm
```

## CI

`.github/workflows/ci.yml`：Python 3.9-3.12 矩阵跑测试 + doctor + bench 冒烟。

# 部署说明

这是一个前后端分离的 RAG 系统。推荐的线上部署方式是：

1. 后端用 `uvicorn` 常驻运行
2. 前端 `vite build` 后交给 Nginx/Apache/静态站点服务
3. 用反向代理把 `/api/v1` 转发到后端
4. `Milvus`、LLM、Embedding 服务按你的实际环境连接
5. 登录鉴权使用 JWT，首个管理员可用 `BOOTSTRAP_ADMIN_*` 环境变量初始化

## 1. 环境变量

后端复制 `backend/.env.example` 为 `backend/.env`，重点改这些：

- `SERVER_HOST=0.0.0.0`
- `SERVER_PORT=8000`
- `CORS_ORIGINS=["https://your-domain.com"]`
- `DATABASE_URL=sqlite:///./app/data/rag_metadata.db`，或者切到 PostgreSQL
- 如果用 MySQL，改成 `mysql+pymysql://user:password@host:3306/rag_db?charset=utf8mb4`
- `UPLOAD_DIR=./data/uploads`
- `CACHE_DIR=./data/cache`
- `JWT_SECRET_KEY=...`
- `BOOTSTRAP_ADMIN_USERNAME` / `BOOTSTRAP_ADMIN_PASSWORD`
- `MILVUS_HOST` / `MILVUS_PORT`
- `LLM_PROVIDER` 和对应模型配置

前端复制 `frontend/.env.example`，线上同域部署时保持：

- `VITE_API_BASE_URL=/api/v1`

## 2. 后端启动

在 `backend` 目录下：

```bash
python scripts/init_milvus_rag_db.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

健康检查：

- `GET /health`

## 3. 前端构建

在 `frontend` 目录下：

```bash
npm run build
```

构建产物在 `frontend/dist`。

## 4. Nginx 示例

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /var/www/enterprise-knowledge/dist;
    index index.html;

    location /api/v1/ {
        proxy_pass http://127.0.0.1:8000/api/v1/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_buffering off;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host $host;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

## 5. 外网访问检查

- 服务器防火墙放行 80/443
- 后端不要直接裸露到公网，只暴露给反向代理
- `CORS_ORIGINS` 要包含实际域名
- 如果前后端分域，前端的 `VITE_API_BASE_URL` 要改成实际 API 地址
- `Milvus`、LLM Provider、Embedding Model 要能从服务器访问到

## 6. Docker 部署

仓库根目录已经提供：

- `backend/Dockerfile`
- `frontend/Dockerfile`
- `docker-compose.yml`

默认用 SQLite 时，直接：

```bash
docker compose up -d --build
```

如果要启用 MySQL，把后端 `DATABASE_URL` 改成 MySQL 连接串，再执行：

```bash
docker compose --profile mysql up -d --build
```

前端容器会通过 Nginx 把 `/api/v1` 和 `/health` 反代到后端。

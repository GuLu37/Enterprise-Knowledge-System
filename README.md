# 🚀 LangChain RAG 三重检索系统

完整的前后端分离单体项目，基于 LangChain 实现 RAG（检索增强生成）三重检索。

## 📚 项目概述

```
三重检索架构：
┌──────────────────────────────────────┐
│         用户查询                      │
└──────────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         ▼         ▼
┌────────┐┌────────┐┌────────┐
│密集检索││稀疏检索││混合检索│
│(向量)  ││(BM25)  ││(融合)  │
└────────┘└────────┘└────────┘
    │         │         │
    └─────────┼─────────┘
              │
    ┌─────────▼─────────┐
    │  结果融合和排序    │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   LLM 生成回答    │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │     返回结果       │
    └───────────────────┘
```

---

## 📁 项目结构

```
Langchain_RAG_tutorial/
├── backend/                          # 后端 (FastAPI + LangChain)
│   ├── app/
│   │   ├── api/                      # API 层
│   │   ├── core/                     # 核心组件 (LLM, Embedding)
│   │   ├── retrieval/                # 三重检索实现
│   │   ├── rag/                      # RAG 链
│   │   ├── services/                 # 业务逻辑
│   │   ├── storage/                  # 存储层
│   │   ├── tools/                    # LLM 工具
│   │   ├── utils/                    # 工具函数
│   │   ├── main.py                   # 应用入口
│   │   └── config.py                 # 配置管理
│   ├── tests/                        # 测试用例
│   ├── data/                         # 数据存储
│   ├── logs/                         # 日志文件
│   ├── requirements.txt              # Python 依赖
│   ├── .env.example                  # 环境变量模板
│   ├── .gitignore
│   └── README.md                     # 后端文档
│
├── frontend/                         # 前端 (Vue/React)
│   ├── src/
│   │   ├── components/               # 组件
│   │   ├── views/                    # 页面
│   │   ├── services/                 # API 服务
│   │   ├── stores/                   # 状态管理
│   │   ├── App.vue
│   │   └── main.js
│   ├── public/
│   ├── package.json
│   ├── vite.config.js
│   ├── .gitignore
│   └── README.md                     # 前端文档
│
└── README.md                         # 项目主文档
```

---

## 🎯 核心特性

### 1. 三重检索融合
- **密集检索 (Dense)** - 基于向量相似度，语义理解
- **稀疏检索 (Sparse)** - 基于 BM25，关键词匹配
- **混合检索 (Hybrid)** - 动态权重融合两种结果

### 2. 多 LLM 支持
- Ollama (本地开源模型)
- OpenAI (GPT-3.5, GPT-4)
- DeepSeek
- Anthropic Claude
- 支持扩展其他模型

### 3. 多向量库支持
- Milvus (高性能向量库)
- Chroma (轻量级选项)
- Pinecone (云服务)

### 4. 完整的文档处理
- PDF, Word, PowerPoint, Excel
- TXT, Markdown
- 自动分块和向量化
- 元数据管理

### 5. 灵活的 Agent 框架
- LangGraph 支持
- 工具调用
- 状态管理
- 多步骤工作流

---

## 🚀 快速开始

### 前置要求

- Python 3.9+
- Node.js 18+
- Ollama 或 其他 LLM
- Milvus 向量数据库
- Docker (可选)

### 后端启动

```bash
cd backend

# 1. 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境
cp .env.example .env
# 编辑 .env 文件

# 4. 启动服务
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 前端启动

```bash
cd frontend

# 1. 安装依赖
npm install

# 2. 配置 API 地址 (.env)
VITE_API_URL=http://localhost:8000/api/v1

# 3. 启动开发服务器
npm run dev
```

### 完整一键启动 (可选)

```bash
# 启动所有服务 (需要 Docker Compose)
docker-compose up -d
```

---

## 📚 文档

- [后端文档](./backend/README.md) - API 接口、配置、开发指南
- [前端文档](./frontend/README.md) - 组件、页面、集成指南

---

## 🔧 配置说明

### 环境变量

复制 `.env.example` 为 `.env`：

```bash
# 后端
cp backend/.env.example backend/.env

# 前端  
cp frontend/.env.example frontend/.env
```

### 后端关键配置

```bash
# LLM 选择
OLLAMA_MODEL=mistral              # 或使用 OpenAI
OPENAI_API_KEY=sk-xxxxx

# 向量数据库
VECTOR_STORE_TYPE=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 检索参数
DENSE_WEIGHT=0.6
SPARSE_WEIGHT=0.4
SEARCH_TOP_K=5
```

### 前端关键配置

```bash
# API 地址
VITE_API_URL=http://localhost:8000/api/v1

# 模型选择
VITE_DEFAULT_MODEL=mistral
```

---

## 🧪 测试

```bash
# 后端测试
cd backend
pytest tests/

# 前端测试
cd frontend
npm run test
```

---

## 🐳 Docker 部署

### 单个容器

```bash
# 后端
docker build -f backend/Dockerfile -t rag-backend:latest backend/
docker run -d -p 8000:8000 rag-backend:latest

# 前端
docker build -f frontend/Dockerfile -t rag-frontend:latest frontend/
docker run -d -p 5173:5173 rag-frontend:latest
```

### Docker Compose

```bash
docker-compose up -d
```

---

## 📊 性能优化

### 检索优化
- 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`
- 优化 `DENSE_WEIGHT` 和 `SPARSE_WEIGHT`
- 增加向量索引

### 缓存配置
```bash
ENABLE_CACHE=true
CACHE_TTL=3600
```

### 数据库优化
- 定期清理过期数据
- 创建向量索引
- 监控查询性能

---

## 🐛 常见问题

### 1. Ollama 连接失败

```bash
# 检查运行状态
curl http://localhost:11434/api/tags

# 拉取模型
ollama pull mistral
ollama pull nomic-embed-text
```

### 2. Milvus 连接失败

```bash
# Docker 启动
docker run -d --name milvus \
  -p 19530:19530 \
  milvusdb/milvus:latest
```

### 3. 前端无法访问后端

- 检查 `VITE_API_URL` 配置
- 确保后端已启动
- 检查 CORS 配置

### 4. 内存不足

- 减少 `CHUNK_SIZE`
- 减少 `SEARCH_TOP_K`
- 分批处理文件

---

## 📈 监控和日志

### 后端日志

```bash
# 日志位置
backend/logs/app.log

# 日志级别配置
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

### 健康检查

```bash
# 后端
curl http://localhost:8000/health

# 前端 (如果启用)
curl http://localhost:5173/health
```

---

## 🔒 安全建议

1. **环境变量管理**
   - 不提交 `.env` 到 Git
   - 使用 `.env.example` 作为模板
   - 生产环境使用密钥管理服务

2. **API 安全**
   - 启用认证 (JWT)
   - HTTPS 加密
   - 速率限制

3. **数据安全**
   - 定期备份
   - 加密敏感数据
   - 访问控制

---

## 🚀 生产部署

### 后端

```bash
# 构建 Docker 镜像
docker build -t rag-backend:v1.0 backend/

# 运行容器
docker run -d \
  -p 8000:8000 \
  -v ./data:/app/data \
  -e DEBUG=false \
  --restart unless-stopped \
  rag-backend:v1.0
```

### 前端

```bash
# 构建生产版本
npm run build

# 使用 Nginx 部署
docker run -d \
  -p 80:80 \
  -v ./dist:/usr/share/nginx/html \
  nginx:alpine
```

---

## 📚 参考资源

### 官方文档
- [LangChain](https://python.langchain.com)
- [FastAPI](https://fastapi.tiangolo.com)
- [Vue 3](https://vuejs.org)
- [Milvus](https://milvus.io)

### 教程和博客
- RAG 入门指南
- 向量数据库最佳实践
- LLM 应用开发

---

## 🤝 贡献指南

欢迎贡献！请：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

---

## 📄 许可证

MIT License - 详见 [LICENSE](./LICENSE)

---

## 👥 作者

项目维护者：[Your Name]

---

## 📞 联系方式

- 📧 Email: your.email@example.com
- 🐛 Issue: GitHub Issues
- 💬 Discussion: GitHub Discussions

---

## 🎓 学习资源

本项目适合用于学习：
- LangChain 框架
- RAG 系统设计
- 向量数据库应用
- FastAPI Web 开发
- 前后端分离架构

---

**最后更新**: 2024-07-26

**版本**: 1.0.0

**状态**: 开发中 🔄

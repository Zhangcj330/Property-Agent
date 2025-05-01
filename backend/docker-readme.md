# Docker 部署指南

本文档提供了如何使用 Docker 和 Docker Compose 部署后端服务的详细说明。

## 前提条件

- 安装 [Docker](https://docs.docker.com/get-docker/)
- 安装 [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)
- 准备好 `Firebase_key.json` 文件（用于Firebase服务认证）
- 创建 `.env` 文件（包含必要的环境变量）

## 环境变量设置

在 backend 目录创建 `.env` 文件，包含以下必要的环境变量：

```
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
BASE_URL=https://generativelanguage.googleapis.com/

# Firebase配置
FIREBASE_PROJECT_ID=your_firebase_project_id

# 其他可选配置
# LOG_LEVEL=INFO
```

## 部署步骤

1. **准备Firebase密钥**

   确保 `Firebase_key.json` 文件位于 backend 目录中。该文件将被挂载到容器中，而不是打包到镜像内。

2. **构建和启动服务**

   ```bash
   cd backend
   docker-compose up -d
   ```

   这将构建Docker镜像并以守护进程模式启动服务。

3. **验证服务状态**

   ```bash
   # 检查容器状态
   docker-compose ps
   
   # 验证健康检查
   curl http://localhost:8000/health
   
   # 测试API接口
   curl http://localhost:8000/api/v1/agent/chat
   ```

## 数据持久化

服务配置使用命名卷 `backend-data` 来持久化存储数据。这确保了容器重启后数据不会丢失。

数据存储位置：
- 容器内：`/app/data`
- 宿主机：由Docker管理（通常在`/var/lib/docker/volumes/`）

## 多环境部署

对于不同环境的部署（开发、测试、生产），您可以：

1. 创建特定环境的配置文件，如 `.env.dev`, `.env.prod`

2. 使用环境变量指定配置文件：

   ```bash
   ENV=prod docker-compose up -d
   ```

   这将使用 `.env.prod` 文件中的配置。

## 监控与日志

查看容器日志：

```bash
# 跟踪实时日志
docker-compose logs -f

# 查看指定行数的日志
docker-compose logs --tail=100
```

## 常见问题排查

1. **容器无法启动**

   检查日志寻找错误信息：
   ```bash
   docker-compose logs backend
   ```

2. **API访问错误**

   确认所有环境变量都正确设置，尤其是API密钥。

3. **Firebase连接问题**

   - 确认Firebase_key.json文件存在且格式正确
   - 验证Firebase项目ID与密钥匹配

## 安全最佳实践

- 敏感凭据（API密钥等）只存储在`.env`文件中，且不应提交到版本控制
- Firebase_key.json以只读模式挂载（`:ro`）
- 使用特定用户而非root运行容器（进阶配置）
- 定期更新基础镜像以修复安全漏洞

## 性能优化提示

- 考虑为数据库服务使用单独的容器
- 针对高访问量场景，使用负载均衡和多个实例
- 对API请求实施速率限制以防止过载 
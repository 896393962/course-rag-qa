# 基于课程资料的智能问答与引用溯源平台

面向高校课程讲义、实验指导书、论文资料和临时上传文档的 RAG 问答平台。项目围绕文档解析、语义检索、元数据过滤、Prompt 构造、引用溯源和检索评估构建完整 AI 应用链路。

## 功能特性

- 构建课程资料知识库，支持 TopK 召回、引用片段返回和答案溯源。
- 基于 `user_id`、`course_id`、`file_id` 做元数据过滤，实现多用户、多课程、多文件的检索隔离。
- 实现“长期课程知识库 + 会话临时文档”双链路，临时文档写入 Redis 并设置 TTL。
- 使用 LangChain LCEL 串联 Retriever、Context Formatter、Prompt Template、Answer Chain 和 Citation Checker。
- 使用 Elasticsearch 建立课程 chunk 索引，支持 BM25 检索和 metadata filter。
- 使用 PostgreSQL 持久化课程资料元数据，记录用户、课程、文件和资料类型。
- 集成 MinerU 文档解析入口，支撑上传资料的文本抽取和 Markdown 化处理。
- 内置检索评估脚本，输出 MRR@K、NDCG@K 等指标。

## 技术栈

- Python 3.10+
- FastAPI / Uvicorn
- LangChain LCEL
- Elasticsearch
- Redis
- PostgreSQL
- MinerU

## 系统流程

```text
用户问题
  -> 元数据过滤
  -> 课程资料召回
  -> 上下文格式化
  -> Prompt 构造
  -> 答案生成
  -> 引用检查
  -> 返回答案、引用、推荐追问和 trace
```

## 快速开始

```powershell
cd projects/course_rag_qa
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8102
```

启动后访问：

- Swagger 文档：http://127.0.0.1:8102/docs
- 健康检查：http://127.0.0.1:8102/health
- 集成状态：http://127.0.0.1:8102/integrations/status

## API 示例

### 课程资料问答

```http
POST /ask
Content-Type: application/json

{
  "question": "RAG 系统为什么要保留引用编号？",
  "user_id": "student_001",
  "course_id": "LLM201",
  "file_id": null,
  "top_k": 5
}
```

### 上传临时文档

```http
POST /documents/temporary
Content-Type: application/json

{
  "title": "实验报告要求",
  "content": "报告需要解释 RAG 的引用溯源设计。",
  "user_id": "student_001",
  "course_id": "LLM201",
  "ttl_seconds": 7200
}
```

### 检索评估

```http
GET /metrics/retrieval?top_k=5
```

命令行运行：

```powershell
python -m app.evaluation
```

## 环境变量

| 变量 | 说明 | 示例 |
| --- | --- | --- |
| `ELASTICSEARCH_URL` | Elasticsearch 服务地址 | `http://127.0.0.1:9200` |
| `REDIS_URL` | Redis 服务地址 | `redis://127.0.0.1:6380/0` |
| `POSTGRES_DSN` | PostgreSQL 连接串 | `postgresql://user:pass@127.0.0.1:5432/db` |
| `USE_MINERU` | MinerU 文档解析开关 | `true` |

## 项目结构

```text
app/
  main.py          # FastAPI 入口
  workflow.py      # RAG 工作流
  retriever.py     # 本地知识库和检索逻辑
  integrations.py  # Redis / Elasticsearch / PostgreSQL / MinerU 集成
  evaluation.py    # 检索评估脚本
  schemas.py       # API 数据模型
data/
  sample_documents.json
  eval_queries.json
```

## 设计重点

项目按照课程资料问答平台的完整链路组织：资料入库、检索过滤、上下文构造、答案生成、引用检查和效果评估分层实现。检索侧强调 metadata filter 与引用溯源，服务侧通过 FastAPI 暴露问答、临时文档上传、集成状态和评估指标接口。

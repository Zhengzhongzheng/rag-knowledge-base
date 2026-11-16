# RAG 知识库问答

一个最小可运行的 RAG（检索增强生成）知识库问答项目。

## 它做什么

读取 `docs/` 目录下的 Markdown / 文本文件，切成文本块。
启动时加载本地中文 Embedding 模型（首次会下载约 100MB），同时建立 BM25 索引和向量索引；
你提问时，分别跑两路检索，再用 RRF 融合成最终 Top-3，交给 DeepSeek 回答并标注来源。

## RAG 流程

```
文档 → 切块 → 建索引 → 检索 Top-K → 拼接 Prompt → 大模型回答
```

## 首次运行

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item ..\ai-first-step\.env .env
```

`.env` 至少需要 DeepSeek Key；Embedding 模型可以用默认值：

```text
DEEPSEEK_API_KEY=sk-你的DeepSeek key
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

首次运行会下载本地向量模型。`run.bat` 已设置 `HF_ENDPOINT=https://hf-mirror.com`，
国内网络下载更稳定；如果模型加载失败，程序会自动回退到 BM25。

然后双击 `run.bat`，或在终端执行：

```powershell
run.bat
```

## 可以问什么

- 我 11 月应该做什么？
- AI 应用开发面试常问什么？
- RAG 的完整流程是什么？
- 如何提升检索效果？

## 当前版本说明

这一版支持三种检索，向量部分使用本地模型 `BAAI/bge-small-zh-v1.5`（512 维）：

- BM25：关键词检索，轻量、快，但对同义表达不敏感。
- 向量检索：语义检索，能理解“录用”和“突围”这类不同措辞的相近含义。
- 混合检索：用 RRF 把两路结果融合，兼顾关键词和语义。

每次提问会同时打印 BM25、向量、混合三组结果，方便对比效果。

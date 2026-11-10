"""RAG 知识库问答：BM25 + Embedding 混合检索。"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

import jieba
import numpy as np
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from rank_bm25 import BM25Plus

jieba.setLogLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 3
EMBEDDING_BATCH_SIZE = 16
RRF_K = 60

load_dotenv(BASE_DIR / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    sys.exit(
        "未找到 DEEPSEEK_API_KEY。\n"
        "请把 ai-first-step/.env 复制到本项目根目录，或按 .env.example 新建。"
    )

chat_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
embedding_client = (
    OpenAI(
        api_key=SILICONFLOW_API_KEY,
        base_url="https://api.siliconflow.cn/v1",
    )
    if SILICONFLOW_API_KEY
    else None
)


def tokenize(text: str) -> list[str]:
    """中文分词，并过滤纯标点和空白。"""
    words = jieba.lcut(text)
    return [
        word.strip().lower()
        for word in words
        if word.strip() and re.search(r"[\w\u4e00-\u9fff]", word)
    ]


def read_documents() -> list[tuple[str, str]]:
    documents: list[tuple[str, str]] = []
    for path in sorted(DOCS_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        documents.append((path.name, path.read_text(encoding="utf-8")))

    if not documents:
        sys.exit(f"没有找到文档，请把 .md 或 .txt 文件放进：{DOCS_DIR}")
    return documents


def split_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """先按段落切，再把超长段落按字符切，并保留重叠。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + size])
                start += size - overlap
            continue

        if len(current) + len(paragraph) + 1 <= size:
            current = f"{current}\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def build_chunks() -> list[dict]:
    chunks: list[dict] = []
    for source, text in read_documents():
        for chunk_id, chunk_text in enumerate(split_text(text)):
            chunks.append(
                {"source": source, "chunk_id": chunk_id, "text": chunk_text}
            )
    return chunks


def build_bm25(chunks: list[dict]) -> BM25Plus:
    return BM25Plus([tokenize(chunk["text"]) for chunk in chunks])


def retrieve_bm25(
    question: str,
    chunks: list[dict],
    bm25: BM25Plus,
    top_k: int = TOP_K,
):
    scores = bm25.get_scores(tokenize(question))
    ranked_indices = sorted(
        range(len(scores)), key=lambda index: scores[index], reverse=True
    )[:top_k]
    return [(chunks[index], float(scores[index])) for index in ranked_indices]


def embed_texts(texts: list[str]) -> np.ndarray:
    """调用 SiliconFlow 的 Embedding API，并做 L2 归一化。"""
    if embedding_client is None:
        raise RuntimeError("未配置 SILICONFLOW_API_KEY")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        vectors.extend([item.embedding for item in response.data])

    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-10)


def build_embedding_index(chunks: list[dict]) -> np.ndarray:
    return embed_texts([chunk["text"] for chunk in chunks])


def retrieve_vector(
    question: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    top_k: int = TOP_K,
):
    query_vector = embed_texts([question])[0]
    scores = embeddings @ query_vector
    ranked_indices = np.argsort(-scores)[:top_k]
    return [
        (chunks[int(index)], float(scores[int(index)]))
        for index in ranked_indices
    ]


def rrf_fuse(result_lists: list[list[tuple[dict, float]]], top_k: int = TOP_K):
    """Reciprocal Rank Fusion：把多路检索结果按排名融合。"""
    scores: dict[tuple[str, int], float] = {}
    chunk_map: dict[tuple[str, int], dict] = {}

    for results in result_lists:
        for rank, (chunk, _) in enumerate(results, start=1):
            key = (chunk["source"], chunk["chunk_id"])
            chunk_map[key] = chunk
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [(chunk_map[key], score) for key, score in ranked]


def build_prompt(question: str, hits) -> str:
    context = "\n\n".join(
        f"[{number}] 来源：{chunk['source']}（第 {chunk['chunk_id'] + 1} 块）\n"
        f"{chunk['text']}"
        for number, (chunk, _) in enumerate(hits, start=1)
    )
    return f"""你是一个严谨的知识库问答助手。请只根据下面的资料回答问题。
如果资料中没有答案，请明确说“资料中没有找到”，不要编造。
回答尽量简洁，并在每条关键结论后用 [来源: 文件名 第X块] 标注依据。
只能引用上面列出的资料，不要编造来源。

资料：
{context}

问题：{question}
"""


def extract_cited_sources(text: str, hits) -> list[str]:
    """只保留模型真正引用过的来源；解析失败时回退到检索结果。"""
    available = {chunk["source"] for chunk, _ in hits}
    cited = re.findall(r"\[来源[:：]\s*([^\]]+?)\]", text)
    names = []
    for item in cited:
        name = item.strip().split()[0]
        if name in available:
            names.append(name)
    return sorted(set(names)) or sorted(available)


def call_chat(prompt: str) -> str:
    response = None
    for attempt in range(1, 4):
        try:
            response = chat_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            break
        except (APIConnectionError, APITimeoutError) as exc:
            if attempt == 3:
                raise
            wait = 2**attempt
            print(f"网络异常（第 {attempt} 次），{wait} 秒后重试...")
            time.sleep(wait)
        except APIStatusError as exc:
            if exc.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                wait = 2**attempt
                print(f"服务繁忙（{exc.status_code}），{wait} 秒后重试...")
                time.sleep(wait)
            else:
                raise

    if response is None:
        raise RuntimeError("多次重试后仍没有收到响应")
    return response.choices[0].message.content


def answer(question: str, hits) -> str:
    if not hits or hits[0][1] <= 0:
        return "资料中没有找到相关内容。"

    text = call_chat(build_prompt(question, hits))
    sources = extract_cited_sources(text, hits)
    return f"{text}\n\n参考来源：{', '.join(sources)}"


def print_results(title: str, hits) -> None:
    print(f"\n[{title}]")
    for chunk, score in hits:
        print(
            f"  - {chunk['source']} 第 {chunk['chunk_id'] + 1} 块"
            f"（score={score:.4f}）"
        )


def main() -> None:
    chunks = build_chunks()
    bm25 = build_bm25(chunks)
    embeddings: np.ndarray | None = None

    if embedding_client is not None:
        print(f"正在构建向量索引（模型：{EMBEDDING_MODEL}）...")
        try:
            embeddings = build_embedding_index(chunks)
            print(f"向量索引已构建：{embeddings.shape[0]} 条，维度 {embeddings.shape[1]}")
        except Exception as exc:
            print(f"向量索引构建失败，回退到 BM25：{exc}")
    else:
        print("未配置 SILICONFLOW_API_KEY，当前只使用 BM25 检索。")

    print(f"知识库已加载：{len(chunks)} 个文本块，来自 {DOCS_DIR}")
    print("输入问题开始提问，输入 exit 退出。\n")

    while True:
        question = input("你：").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        bm25_hits = retrieve_bm25(question, chunks, bm25)
        print_results("BM25 检索", bm25_hits)

        vector_hits: list[tuple[dict, float]] = []
        if embeddings is not None:
            try:
                vector_hits = retrieve_vector(question, chunks, embeddings)
                print_results("向量检索", vector_hits)
            except Exception as exc:
                print(f"\n向量检索失败，继续使用 BM25：{exc}")
                embeddings = None

        final_hits = rrf_fuse([bm25_hits, vector_hits]) if vector_hits else bm25_hits
        print_results("混合检索最终结果", final_hits)

        try:
            print(f"\nAI：{answer(question, final_hits)}\n")
        except Exception as exc:
            print(f"调用失败：{exc}\n")


if __name__ == "__main__":
    main()

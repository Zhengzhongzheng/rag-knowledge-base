"""RAG 知识库的 Streamlit 网页界面。"""

import hashlib
import io

import streamlit as st
from pypdf import PdfReader

import main

st.set_page_config(
    page_title="AI 求职知识库",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource(show_spinner="正在加载知识库和本地向量模型...")
def load_engine():
    chunks = main.build_chunks()
    bm25 = main.build_bm25(chunks)
    main.embedding_model = main.TextEmbedding(main.EMBEDDING_MODEL)
    embeddings = main.build_embedding_index(chunks)
    return chunks, bm25, embeddings


@st.cache_data(show_spinner=False)
def extract_uploaded_text(file_name: str, file_bytes: bytes) -> str:
    if file_name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return file_bytes.decode("utf-8", errors="ignore")


def build_uploaded_chunks(uploaded_files):
    chunks = []
    failed_files = []

    for uploaded_file in uploaded_files:
        try:
            text = extract_uploaded_text(uploaded_file.name, uploaded_file.getvalue())
        except Exception:
            failed_files.append(f"{uploaded_file.name}（解析失败）")
            continue

        if not text.strip():
            failed_files.append(f"{uploaded_file.name}（没有提取到文字）")
            continue

        for chunk_id, chunk_text in enumerate(main.split_text(text)):
            chunks.append(
                {
                    "source": uploaded_file.name,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )

    return chunks, failed_files


def files_signature(uploaded_files) -> str:
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


try:
    base_chunks, base_bm25, base_embeddings = load_engine()
except Exception as exc:
    st.error(f"知识库加载失败：{exc}")
    st.stop()

st.title("🤖 我的 AI 求职知识库")
st.caption("上传 PDF / TXT / MD，基于 BM25 + 中文向量检索 + DeepSeek 回答")

with st.sidebar:
    st.header("知识库")
    uploaded_files = st.file_uploader(
        "上传文档（PDF / TXT / MD）",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="上传的文档只在本次会话中有效，不会写入服务器磁盘。",
    )

    if uploaded_files:
        signature = files_signature(uploaded_files)
        if (
            st.session_state.get("upload_signature") != signature
            or "active_chunks" not in st.session_state
        ):
            with st.spinner("正在解析上传文档并重建索引..."):
                uploaded_chunks, failed_files = build_uploaded_chunks(uploaded_files)
                active_chunks = base_chunks + uploaded_chunks
                active_bm25 = main.build_bm25(active_chunks)
                active_embeddings = main.build_embedding_index(active_chunks)

                st.session_state.upload_signature = signature
                st.session_state.active_chunks = active_chunks
                st.session_state.active_bm25 = active_bm25
                st.session_state.active_embeddings = active_embeddings
                st.session_state.failed_files = failed_files

        active_chunks = st.session_state.active_chunks
        active_bm25 = st.session_state.active_bm25
        active_embeddings = st.session_state.active_embeddings

        st.caption("已上传：" + "、".join(file.name for file in uploaded_files))
        if st.session_state.get("failed_files"):
            st.warning(
                "这些文件没有加入知识库："
                + "、".join(st.session_state.failed_files)
            )
    else:
        st.session_state.upload_signature = ""
        active_chunks = base_chunks
        active_bm25 = base_bm25
        active_embeddings = base_embeddings

    st.metric("文本块", len(active_chunks))
    st.metric("向量维度", active_embeddings.shape[1])
    st.write("Embedding 模型：`BAAI/bge-small-zh-v1.5`")
    st.write("生成模型：`deepseek-chat`")
    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("查看检索来源"):
                for chunk in message["sources"]:
                    st.markdown(
                        f"**{chunk['source']} · 第 {chunk['chunk_id'] + 1} 块**"
                    )
                    st.text(chunk["text"][:400])

question = st.chat_input("问关于上传文档、求职规划、RAG 项目的问题...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    bm25_hits = main.retrieve_bm25(question, active_chunks, active_bm25)
    vector_hits = main.retrieve_vector(question, active_chunks, active_embeddings)
    final_hits = main.rrf_fuse([bm25_hits, vector_hits])

    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成回答..."):
            try:
                response = main.answer(question, final_hits)
            except Exception as exc:
                response = f"调用失败：{exc}"
        st.markdown(response)

        with st.expander("查看检索来源"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**BM25 检索**")
                for chunk, score in bm25_hits:
                    st.write(
                        f"{chunk['source']} · 第 {chunk['chunk_id'] + 1} 块"
                        f"（{score:.2f}）"
                    )
            with col2:
                st.markdown("**向量检索**")
                for chunk, score in vector_hits:
                    st.write(
                        f"{chunk['source']} · 第 {chunk['chunk_id'] + 1} 块"
                        f"（{score:.2f}）"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "sources": [chunk for chunk, _ in final_hits],
        }
    )

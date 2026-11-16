"""RAG 知识库的 Streamlit 网页界面。"""

import streamlit as st

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


try:
    chunks, bm25, embeddings = load_engine()
except Exception as exc:
    st.error(f"知识库加载失败：{exc}")
    st.stop()

st.title("🤖 我的 AI 求职知识库")
st.caption("BM25 + 中文向量检索 + DeepSeek 生成，回答会标注来源")

with st.sidebar:
    st.header("知识库")
    st.metric("文本块", len(chunks))
    st.metric("向量维度", embeddings.shape[1])
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

question = st.chat_input("问关于求职规划、RAG 项目、面试准备的问题...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    bm25_hits = main.retrieve_bm25(question, chunks, bm25)
    vector_hits = main.retrieve_vector(question, chunks, embeddings)
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

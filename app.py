import re
import pickle
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
st.set_page_config(page_title="Ask the Book", page_icon="📖")

tok = lambda s: re.findall(r"\w+", s.lower())


# ---------- load everything once (cached across reruns) ----------
@st.cache_resource(show_spinner="Loading the book index...")
def load_resources():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name="got_local",
        embedding_function=embeddings,
        persist_directory="./chroma_db_local",
    )
    bm25 = BM25Okapi([tok(c.page_content) for c in chunks])

    by_page = defaultdict(list)
    for c in chunks:
        by_page[c.metadata["page"]].append(c)
    order = {id(c): i for i, c in enumerate(chunks)}
    chapters = list(dict.fromkeys(c.metadata["chapter"] for c in chunks))  # book order

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    return chunks, vectorstore, bm25, by_page, order, chapters, llm


chunks, vectorstore, bm25, by_page, order, chapters, llm = load_resources()


# ---------- retrieval ----------
def dedupe(docs):
    seen, out = set(), []
    for d in docs:
        key = (d.metadata["page"], d.page_content[:60])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def keyword_search(query, k=5, chapter=None):
    scores = bm25.get_scores(tok(query))
    idx = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    if chapter:
        idx = [i for i in idx if chunks[i].metadata["chapter"] == chapter]
    return [chunks[i] for i in idx[:k]]


def hybrid_search(query, k=5, chapter=None):
    flt = {"chapter": chapter} if chapter else None
    docs = vectorstore.similarity_search(query, k=k, filter=flt)
    docs += keyword_search(query, k=k, chapter=chapter)
    return dedupe(docs)


def expand_queries(question, n=3):
    prompt = f"""You help search the novel A Game of Thrones.
For the question below, write:
- {n} alternative search queries, one per line, each starting with "Q: ".
  Use names and events the book itself would use.
- one line starting with "K: " with 6-10 single keywords likely to appear in the
  passage that answers it, including synonyms and background events
  (example: for a question about why someone is hated, K: grudge, betrayal, insult, feud, revenge).

Question: {question}"""
    text = llm.invoke(prompt).content[0]["text"]
    lines = text.splitlines()
    queries = [l[2:].strip() for l in lines if l.startswith("Q:")][:n]
    keywords = " ".join(l[2:].strip() for l in lines if l.startswith("K:"))
    return [question] + queries, keywords


def retrieve(question, k=3, chapter=None):
    queries, keywords = expand_queries(question)
    hits = []
    for q in queries:
        hits += hybrid_search(q, k=k, chapter=chapter)
    if keywords:
        hits += keyword_search(keywords, k=6, chapter=chapter)
    return dedupe(hits)


def with_neighbors(docs, after=1):
    seen, out = set(), []
    for d in docs:
        p0, ch = d.metadata["page"], d.metadata["chapter"]
        for p in range(p0, p0 + after + 1):
            for c in by_page.get(p, []):
                if c.metadata["chapter"] == ch:
                    key = (p, c.page_content[:60])
                    if key not in seen:
                        seen.add(key)
                        out.append(c)
    return sorted(out, key=lambda c: order[id(c)])


# ---------- answering ----------
def format_sources(docs):
    """Group pages by chapter and compress runs: 'Chapter 8: Bran, p. 83-89, 95'."""
    pages_by_chapter = defaultdict(set)
    for d in docs:
        pages_by_chapter[d.metadata["chapter"]].add(d.metadata["page"])

    lines = []
    for chapter, pages in pages_by_chapter.items():  # docs are in book order
        pages = sorted(pages)
        ranges, start = [], pages[0]
        prev = start
        for p in pages[1:]:
            if p == prev + 1:
                prev = p
                continue
            ranges.append((start, prev))
            start = prev = p
        ranges.append((start, prev))
        text = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)
        lines.append(f"**{chapter}** — p. {text}")
    return lines


def ask(question, k=3, after=1, chapter=None):
    hits = retrieve(question, k=k, chapter=chapter)
    docs = with_neighbors(hits, after=after)
    context = "\n\n".join(
        f"[{d.metadata['chapter']}, p.{d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )
    prompt = f"""Answer the question using only the excerpts below from A Game of Thrones.
If the excerpts don't contain the answer, say so instead of guessing.
Lead with a direct answer in one or two sentences, then add key supporting details
with chapter and page. Don't list every minor detail.

Excerpts:
{context}

Question: {question}"""
    answer = llm.invoke(prompt).content[0]["text"]
    return answer, format_sources(docs)


def summarize_chapter(chapter_name):
    docs = [c for c in chunks if c.metadata["chapter"] == chapter_name]  # book order
    context = "\n\n".join(f"[p.{d.metadata['page']}]\n{d.page_content}" for d in docs)
    prompt = f"""Summarize what happens in this chapter of A Game of Thrones, in order.
Mention page numbers for the main events.

{context}"""
    return llm.invoke(prompt).content[0]["text"]


# ---------- UI ----------
st.title("📖 Ask the Book")
st.caption(
    "A Game of Thrones. Answers come only from passages retrieved from the book. "
    "If it says it can't find something, the passage may simply not have been retrieved."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "summaries" not in st.session_state:
    st.session_state.summaries = {}

with st.sidebar:
    st.header("Options")
    choice = st.selectbox("Limit questions to a chapter", ["All chapters"] + chapters)
    chapter = None if choice == "All chapters" else choice

    if st.button("Summarize this chapter", disabled=chapter is None):
        try:
            with st.spinner("Summarizing..."):
                if chapter not in st.session_state.summaries:
                    st.session_state.summaries[chapter] = summarize_chapter(chapter)
            st.session_state.messages.append(
                {"role": "assistant",
                 "content": f"**Summary of {chapter}**\n\n{st.session_state.summaries[chapter]}",
                 "sources": [f"**{chapter}**"]}
            )
        except Exception as e:
            st.error(f"Could not summarize: {e}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for line in m["sources"]:
                    st.markdown(line)

if question := st.chat_input("Ask something about the book"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the book..."):
                answer, sources = ask(question, chapter=chapter)
            st.markdown(answer)
            with st.expander("Sources"):
                for line in sources:
                    st.markdown(line)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Something went wrong (a 429 means your Gemini quota ran out): {e}")
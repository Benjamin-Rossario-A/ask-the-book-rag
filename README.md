# Ask the Book

A retrieval-augmented generation (RAG) app that answers questions about a novel and shows **which chapter and page** each answer came from. Built with LangChain, Chroma, local embeddings, Gemini and a Streamlit chat UI. I tested it on *A Game of Thrones*.


![App screenshot](screenshot.png)
## Features

- Answers grounded in retrieved passages, with chapter and page citations
- Says so when the retrieved excerpts don't contain the answer, instead of guessing
- Chapter dropdown to limit questions to one chapter
- One-click chapter summaries (the whole chapter is sent to the model, no search)
- Hybrid retrieval: embeddings + keyword (BM25) + LLM query rewriting

## How it works

1. **Load and tag.** `PyPDFLoader` reads the PDF page by page. Front matter and map pages are dropped, and every page is tagged with its chapter by detecting the chapter headings. Page numbers are stored 1-based to match a PDF viewer.
2. **Chunk.** `RecursiveCharacterTextSplitter` (1,000 characters, 150 overlap) produces about 2,100 chunks. Page and chapter metadata is carried onto every chunk.
3. **Embed.** Chunks are embedded locally with `BAAI/bge-small-en-v1.5` and stored in Chroma.
4. **Retrieve.** The LLM rewrites the question into three alternative queries plus a line of likely keywords. Each query runs an embedding search and a BM25 keyword search, the results are merged, and the following page of each hit (same chapter) is added.
5. **Answer.** Gemini receives the labelled excerpts and a prompt that tells it to answer only from them, lead with a direct answer, and cite chapter and page.

## What I learned while building it

- **Plain top-k search missed the answer.** For "Why does Bran fall from the tower?", the matching text was the scene *before* the cause. The cause sat on the next page. Adding the following pages of each hit fixed it.
- **MMR made it worse here.** Diversity filtering dropped the most useful chunks because they resembled each other. I went back to plain similarity.
- **Embeddings can miss a single sentence.** "Why is Theon Greyjoy at Winterfell?" is answered by one line inside a flashback. It did not appear in the top 50 embedding results, but keyword search ranked it second for the right words. Query rewriting (to supply those words) plus BM25 fixed it, though results still vary a little between runs because the rewrites come from an LLM.
- **Free-tier limits shaped the design.** Gemini's free embedding quota (1,000 requests per day) was too small for about 2,100 chunks, so embeddings run locally and Gemini is only used for rewriting and answering.

## Tech stack

Python, Streamlit, LangChain, Chroma, sentence-transformers (`bge-small-en-v1.5`), rank-bm25, Gemini (`gemini-3.5-flash-lite`).

## Setup

You need your own copy of the PDF (it is not included) and a free Gemini API key.

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):

```
GEMINI_API_KEY=your_key_here
```

### Build the index

Put your PDF in the project folder, set `PDF_PATH` in `first-book.ipynb`, and run all cells. This creates `chunks.pkl` and the `chroma_db_local/` folder. Run it once; the embedding cell skips work if the database already exists.

### Run the app

```bash
python -m streamlit run app.py
```

## Project structure

```
app.py               Streamlit chat app (retrieval + answering)
first-book.ipynb     Index builder: load, tag chapters, chunk, embed
requirements.txt
.env.example
```

`chunks.pkl` and `chroma_db_local/` are generated and not committed. They must come from the same run.

## Limitations

- **Chapter detection is specific to this PDF.** It looks for point-of-view headings (PROLOGUE, BRAN, CATELYN, ...) at the top of a page and skips the first pages by number. Another book needs different detection.
- **Each question is independent.** The app shows chat history, but the model does not see earlier turns.
- **"Not found" means "not retrieved".** A refusal can mean the book doesn't say it, or that the right passage wasn't retrieved.
- **Images are ignored.** Maps and illustrations are not indexed.
- **Answers come from a small model.** It can still garble the order of events in a long scene.

## Roadmap

- Conversation memory for follow-up questions
- An evaluation script built from a fixed list of test questions
- Streaming answers
- A standalone `ingest.py` replacing the notebook

## Note on the book

The novel is copyrighted and is not included in this repository. Bring your own copy for personal use.

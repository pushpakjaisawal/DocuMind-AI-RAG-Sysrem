#  DocuMind AI: Multimodal Vision-RAG Pipeline 📄🚀

An enterprise-grade, multimodal Retrieval-Augmented Generation (RAG) system. Unlike traditional text-only RAG, DocuMind preserves visual layout by treating document pages as images, extracting data exhaustively at ingest-time, and utilizing a high-speed LPU for precision generation.

---

## ✨ Features & Core Functions (What it can perform)

- 👁️ **Vision Analysis:** Reads and interprets complex charts, tables, and diagrams directly from PDF pages.
-  **Precision Extraction:** Extracts exact numbers, dates, skills, and metrics verbatim without conversational fluff.
- 🧠 **Conversational Memory:** Remembers previous questions for seamless, context-aware follow-up queries.
- ⚡ **Lightning Fast Queries:** Pre-processes documents at upload, making query responses sub-second.
- 🔄 **Smart Deduplication:** Automatically detects already-indexed documents to save API tokens and compute.
- 🌑 **Sleek Pure-Black UI:** A modern, distraction-free interface with real-time ingestion status tracking.

---

## 🏗️ System Architecture & Data Flow

Standard RAG systems fail on visual documents because they process images at query time, causing massive latency. We solved this by shifting the heavy lifting to the **ingestion phase**.

**Data Flow:** 
`PDF Upload` ➔ `Poppler (Render)` ➔ `NVIDIA Vision (Extract)` ➔ `Jina (Embed)` ➔ `ChromaDB (Store)` 
*(Query Time: `User Prompt` ➔ `Jina (Embed)`  `ChromaDB (Retrieve Top-4)`  `Groq (Generate with History)`)*

| Layer | Technology | Model / Tool | Role |
| :--- | :--- | :--- | :--- |
| 🎨 **Frontend** | Streamlit | Custom Pure-Black UI | State management, conversational history, and live status tracking. |
| ️ **Backend** | FastAPI | Pydantic, Uvicorn | Asynchronous REST API, file handling, and deduplication logic. |
| 👁️ **Vision** | NVIDIA NIM | `meta/llama-3.2-11b-vision-instruct` | Ingest-time exhaustive OCR and data extraction (charts, tables, text). |
|  **Embedding** | Jina AI | `jina-embeddings-v5-omni-small` | Batch-embeds extracted text into a shared 1024-dim vector space. |
| ️ **Vector DB** | ChromaDB | Local Persistent Client | Stores vectors and metadata with cosine similarity. |
| 🤖 **Generation** | Groq | `openai/gpt-oss-120b` | Ultra-low latency, context-aware generation with conversational memory. |

---

## 🚀 Getting Started

### 🛠️ 1. Prerequisites
- Python 3.9+
- **Poppler** (Required for `pdf2image` to render PDFs):
  - **Ubuntu/Debian**: `sudo apt-get install poppler-utils`
  - **macOS**: `brew install poppler`
  - **Windows**: Download [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases), extract, and add the `Library\bin` folder to your System Environment Variables `PATH`.

### 📥 2. Installation
```bash
git clone https://github.com/[YOUR_USERNAME]/documind-ai.git
cd documind-ai

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # On Mac/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
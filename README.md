# 🏟️ StapuBox Venue Assistant — Grounded RAG Application

> **Built for the AI Product Engineer Intern role**
>
> A production-ready Retrieval-Augmented Generation (RAG) application that answers
> natural-language queries about sports venues with strict JSON responses and
> traceable citations — eliminating hallucinations by grounding every answer in a
> curated knowledge base.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **Retrieval-Augmented Generation** | Every answer is grounded in the venue knowledge base — the model cannot hallucinate facts that aren't in the source documents. |
| **Vector Search (FAISS + `text-embedding-004`)** | Documents are embedded with Google's `text-embedding-004` model and indexed in a FAISS inner-product index for sub-millisecond cosine-similarity retrieval. |
| **Strict Schema Validation (Pydantic + `gemini-2.5-flash`)** | Gemini's `response_schema` parameter enforces a Pydantic model at generation time, guaranteeing a valid `{"answer": "...", "citations": [...]}` JSON output on every call. |
| **Traceable Citations** | Each response includes an array of citations with the exact source section and a verbatim excerpt, so users can verify every claim. |
| **Modular Single-File Architecture** | 10 clearly separated modules (config, models, chunker, embedder, FAISS manager, orchestrator, generator, UI) in one self-contained `app.py`. |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐     ┌──────────────────────┐
│  Embed Query     │────▶│  FAISS Index          │
│  (RETRIEVAL_QUERY│     │  (30 chunks, 768-dim) │
│   text-embed-004)│     └──────────┬───────────┘
└─────────────────┘                │ top-5 chunks
                                   ▼
                    ┌──────────────────────────┐
                    │  Gemini 2.5 Flash        │
                    │  + system prompt         │
                    │  + Pydantic schema       │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  JSON Response           │
                    │  { answer, citations[] } │
                    └──────────────────────────┘
```

### Knowledge Base — 3 Venues

| Venue | Facilities |
|---|---|
| **StapuBox Arena Central** | Tennis courts · Basketball courts · Swimming pool |
| **Greenfield Sports Complex** | Cricket nets · Football pitches · Badminton halls |
| **Summit Athletic Club** | Squash courts · Gym & fitness center · Indoor volleyball |

Each venue includes: operating hours, sport-specific rules, pricing, dress code, equipment rental, booking policies, and membership plans.

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.10+**
- A **Google Gemini API key** — get one at [aistudio.google.com](https://aistudio.google.com/)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/stapubox-venue-assistant.git
cd stapubox-venue-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `google-genai`, `faiss-cpu`, `numpy`, and `pydantic`.

### 3. Configure the API Key

Copy the example secrets file and add your key:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your-actual-gemini-api-key"
```

**Alternative methods:**

- **Environment variable:** `export GOOGLE_API_KEY="your-key"` (Linux/macOS) or `$env:GOOGLE_API_KEY="your-key"` (PowerShell)
- **Sidebar input:** Paste the key directly into the app's sidebar at runtime

### 4. Run the App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**. Type a question or click an example in the sidebar.

---

## 📂 Project Structure

```
stapubox-venue-assistant/
├── .gitignore
├── .streamlit/
│   ├── secrets.toml.example    # Template — copy to secrets.toml
│   └── secrets.toml            # Your actual key (git-ignored)
├── app.py                      # Single-file Streamlit RAG application
├── requirements.txt            # Python dependencies
├── venue_knowledge_base.txt    # Knowledge base (3 venues × 10 sections)
└── README.md                   # This file
```

---

## 💬 Example Queries

- *"What are the tennis court rules at StapuBox Arena Central?"*
- *"When is Summit Athletic Club open on weekends?"*
- *"What is the dress code for the swimming pool?"*
- *"How much does badminton court booking cost at Greenfield?"*
- *"Can I rent squash equipment at Summit Athletic Club?"*
- *"What is the cancellation policy at Greenfield Sports Complex?"*

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Embeddings** | Google `text-embedding-004` via `google-genai` SDK |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| **Generation** | Google `gemini-2.5-flash` with structured output |
| **Validation** | [Pydantic](https://docs.pydantic.dev/) v2 |

---

## 📄 License

MIT

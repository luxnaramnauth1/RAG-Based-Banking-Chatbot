# RAG-Based-Banking-Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers customer questions about **accounts, loans, credit cards, fees, security, and digital banking**, grounded entirely in a bank's own policy knowledge base and powered by an **open-source LLM** (no paid API keys required).

Built with **Python, LlamaIndex, and Hugging Face Transformers**.

<img width="924" height="838" alt="2027" src="https://github.com/user-attachments/assets/e612df4d-4782-4849-aa58-98cac936a33d" />

## Why this project

Off-the-shelf LLMs will confidently invent interest rates, fees, and policies they were never told about — a serious problem in a regulated domain like banking. This project demonstrates a RAG pipeline that:

- Retrieves the most relevant policy snippets from a knowledge base for every question
- Forces the model to answer **only** from that retrieved context via prompt constraints
- Explicitly refuses to answer when the knowledge base doesn't cover the question, instead of hallucinating
- Returns cited sources alongside every answer, for transparency and auditability

## Architecture

```
User question
     │
     ▼
Embed question (sentence-transformers/all-MiniLM-L6-v2)
     │
     ▼
Vector similarity search over the banking knowledge base
     │
     ▼
Top-k relevant chunks retrieved
     │
     ▼
Chunks + question inserted into a grounded prompt template
     │
     ▼
Open-source LLM (TinyLlama-1.1B-Chat, swappable) generates the answer
     │
     ▼
Answer + cited sources returned to the user
```

## Project structure

```
rag-banking-chatbot/
├── data/
│   └── banking_kb.json        # 26-document banking knowledge base (8 categories)
├── src/
│   ├── config.py               # model + retrieval settings (single place to tune)
│   ├── ingest.py                # builds & persists the vector index
│   └── chatbot.py               # query engine + interactive CLI chat loop
├── RAG_Banking_Chatbot.ipynb  # main walkthrough notebook (recommended entry point)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run it

**Option A — Notebook (recommended for a first look / demo):**

```bash
jupyter notebook RAG_Banking_Chatbot.ipynb
```

Run the cells top to bottom. The first run will download the embedding model (~80MB) and the LLM (~2.2GB) from Hugging Face — this only happens once.

**Option B — Command-line chatbot:**

```bash
python src/ingest.py     # builds the vector index (run once)
python src/chatbot.py    # starts an interactive chat session
```

```
You: How much is the overdraft fee?
Bot: An overdraft fee of $34 is charged per transaction, up to a maximum of 3 fees per day ($102)...
Sources:
  - Overdraft fees (relevance: 0.847)
```

## Configuration

Everything tunable lives in `src/config.py`:

| Setting | Purpose |
|---|---|
| `EMBED_MODEL_NAME` | Hugging Face embedding model used for retrieval |
| `LLM_MODEL_NAME` | Local Hugging Face model used for generation |
| `USE_HF_INFERENCE_API` | Set `True` to call a hosted model via the free HF Inference API instead of downloading weights (good for low-resource machines) |
| `TOP_K` | Number of retrieved chunks passed to the LLM |
| `SYSTEM_PROMPT` | The grounding/anti-hallucination instruction given to the model |

No GPU? Set `USE_HF_INFERENCE_API = True` and export a free token:

```bash
export HUGGINGFACE_API_TOKEN=your_token_here
```

Have a GPU and want better answers? Swap `LLM_MODEL_NAME` for something like `Qwen/Qwen2.5-7B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.3` — no other code changes needed.

## Evaluation approach

- **Retrieval quality:** hit-rate/MRR against a labeled set of (question → correct source) pairs
- **Faithfulness:** does the generated answer only state facts present in the retrieved context?
- **Refusal correctness:** does the bot correctly decline out-of-scope questions rather than hallucinate?

## Possible extensions

- Ingest real PDFs (loan agreements, T&Cs) instead of a flat JSON knowledge base
- Add multi-turn conversation memory
- Add a reranking step (e.g. `bge-reranker`) after initial retrieval for higher precision
- Wrap in a Streamlit/Gradio UI for a live demo link
- Add guardrails/PII filtering before responses reach the user
- Swap local storage for a production vector DB (Chroma, Qdrant, Pinecone)

## Skills demonstrated

RAG system design · vector embeddings & similarity search · open-source LLM integration (Hugging Face) · prompt engineering for factual grounding & hallucination control · Python software structure (config-driven, modular) · domain modeling for a regulated industry (banking)

Face, LlamaIndex). Designed a grounded prompting strategy to prevent hallucinated financial figures and implemented source-citation for auditability.

> Designed and implemented an end-to-end RAG pipeline (chunking → embedding → vector retrieval → constrained generation) in Python using LlamaIndex and Hugging Face Transformers, achieving accurate, source-cited answers over a domain-specific banking knowledge base while explicitly refusing out-of-scope queries to reduce hallucination risk.

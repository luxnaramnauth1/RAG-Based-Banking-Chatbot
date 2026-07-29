"""
Central configuration for the RAG Banking Chatbot.

Swap EMBED_MODEL_NAME or LLM_MODEL_NAME to try different open-source models
from Hugging Face without touching any other code.
"""

import os

# --- Paths -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "banking_kb.json")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")  # persisted vector index

# --- Embedding model -----------------------------------------------------
# Small, fast, runs on CPU. ~80MB download, 384-dim embeddings.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM ------------------------------------------------------------------
# A small open-source instruct model that runs reasonably on CPU.
# For better answer quality on a machine with a GPU, swap to something like
# "Qwen/Qwen2.5-7B-Instruct" or "mistralai/Mistral-7B-Instruct-v0.3".
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Alternative: use the free Hugging Face Inference API instead of downloading
# weights locally (useful on low-resource machines). Set USE_HF_INFERENCE_API
# to True and provide a token via the HUGGINGFACE_API_TOKEN environment
# variable to use this path instead of a locally-loaded model.
USE_HF_INFERENCE_API = False
HF_INFERENCE_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# --- Chunking / retrieval -------------------------------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 3  # number of retrieved chunks passed to the LLM as context

# --- Generation -------------------------------------------------------------
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1  # low temperature -> more factual, less creative answers

SYSTEM_PROMPT = (
    "You are a helpful, precise banking assistant for a retail bank. "
    "Answer the customer's question using ONLY the information given in the "
    "context below. If the answer is not contained in the context, say "
    "'I don't have that information — please contact customer service.' "
    "Do not make up policies, numbers, or fees. Keep answers concise and "
    "cite specific figures (fees, rates, timeframes) from the context when "
    "relevant."
)

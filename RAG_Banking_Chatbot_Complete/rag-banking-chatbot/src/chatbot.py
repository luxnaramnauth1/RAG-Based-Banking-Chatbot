"""
RAG Banking Chatbot — query engine + CLI chat loop.

Run:
    python src/chatbot.py

This loads (or builds) the vector index, wires up a local open-source LLM,
and starts an interactive command-line chat session. Type 'exit' to quit.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from ingest import load_or_build_index

from llama_index.core import Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer


def load_llm():
    """Load either a local Hugging Face model or the HF Inference API client,
    depending on config.USE_HF_INFERENCE_API."""
    if config.USE_HF_INFERENCE_API:
        from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

        token = os.environ.get("HUGGINGFACE_API_TOKEN")
        if not token:
            raise RuntimeError(
                "Set the HUGGINGFACE_API_TOKEN environment variable to use "
                "the Hugging Face Inference API."
            )
        return HuggingFaceInferenceAPI(
            model_name=config.HF_INFERENCE_MODEL,
            token=token,
            temperature=config.TEMPERATURE,
            max_new_tokens=config.MAX_NEW_TOKENS,
        )
    else:
        from llama_index.llms.huggingface import HuggingFaceLLM

        return HuggingFaceLLM(
            model_name=config.LLM_MODEL_NAME,
            tokenizer_name=config.LLM_MODEL_NAME,
            context_window=2048,
            max_new_tokens=config.MAX_NEW_TOKENS,
            generate_kwargs={
                "temperature": config.TEMPERATURE,
                "do_sample": config.TEMPERATURE > 0,
            },
            device_map="auto",
        )


QA_TEMPLATE = PromptTemplate(
    config.SYSTEM_PROMPT
    + "\n\n---------------------\n"
    + "Context:\n{context_str}\n"
    + "---------------------\n\n"
    + "Question: {query_str}\n"
    + "Answer: "
)


def build_chat_engine():
    print("Setting up embedding model + LLM ...")
    index = load_or_build_index()  # also sets Settings.embed_model

    Settings.llm = load_llm()

    query_engine = index.as_query_engine(
        similarity_top_k=config.TOP_K,
        text_qa_template=QA_TEMPLATE,
    )
    return query_engine


def ask(query_engine, question: str):
    response = query_engine.query(question)
    sources = [
        (node.metadata.get("title"), round(node.score, 3))
        for node in response.source_nodes
    ]
    return str(response), sources


def chat_loop():
    query_engine = build_chat_engine()
    print("\n💬 RAG Banking Chatbot ready. Ask about accounts, loans, cards, fees, etc.")
    print("   Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        answer, sources = ask(query_engine, question)
        print(f"\nBot: {answer}\n")
        if sources:
            print("Sources:")
            for title, score in sources:
                print(f"  - {title} (relevance: {score})")
        print()


if __name__ == "__main__":
    chat_loop()

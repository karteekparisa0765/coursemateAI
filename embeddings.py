import os

from langchain_mistralai import MistralAIEmbeddings


EMBEDDING_MODEL_NAME = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")


def get_embeddings() -> MistralAIEmbeddings:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY. Add it to your environment or .env file.")

    return MistralAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=api_key,
    )

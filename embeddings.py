import os
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _resolve_local_model_path() -> str | None:
    hub_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    model_dir = hub_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    ref_file = model_dir / "refs" / "main"

    if not ref_file.exists():
        return None

    snapshot_id = ref_file.read_text(encoding="utf-8").strip()
    snapshot_dir = model_dir / "snapshots" / snapshot_id

    if snapshot_dir.exists():
        return str(snapshot_dir)

    return None


def get_embeddings() -> HuggingFaceEmbeddings:
    local_model_path = _resolve_local_model_path()

    return HuggingFaceEmbeddings(
        model_name=local_model_path or EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": "cpu",
            "local_files_only": local_model_path is not None,
        },
        encode_kwargs={"normalize_embeddings": True},
    )

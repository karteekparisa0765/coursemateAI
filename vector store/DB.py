from rag_pipeline import build_retriever, ingest_pdf


def smoke_test() -> None:
    result = ingest_pdf(
        pdf_path="document loaders/deeplearning.pdf",
        document_name="deeplearning.pdf",
        replace_existing=True,
    )
    print(f"Indexed {result.document_name}: {result.page_count} pages, {result.chunk_count} chunks")

    retriever = build_retriever()
    docs = retriever.invoke("What is deep learning?")

    for doc in docs:
        print(doc.metadata)
        print(doc.page_content[:300])
        print()


if __name__ == "__main__":
    smoke_test()

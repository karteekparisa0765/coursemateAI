from rag_pipeline import ingest_pdf


def main() -> None:
    result = ingest_pdf(
        pdf_path="document loaders/deeplearning.pdf",
        document_name="deeplearning.pdf",
        replace_existing=True,
    )
    print(f"Indexed {result.document_name}: {result.page_count} pages, {result.chunk_count} chunks")


if __name__ == "__main__":
    main()

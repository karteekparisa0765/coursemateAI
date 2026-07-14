from rag_pipeline import answer_question, build_retriever, require_mistral_api_key


def main() -> None:
    require_mistral_api_key()
    build_retriever()

    print("CourseMate CLI is ready.")
    print("Type `0` to exit.")

    while True:
        query = input("You: ").strip()
        if query == "0":
            break
        if not query:
            print("AI: Please enter a question.")
            continue

        answer, _ = answer_question(query)
        print(f"\nAI: {answer}\n")


if __name__ == "__main__":
    main()

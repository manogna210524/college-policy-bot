from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = str(BASE_DIR / "faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not DATA_DIR.exists():
        raise SystemExit("Missing data/ folder. Add policy PDFs inside category subfolders.")

    documents = []
    for category_dir in sorted(DATA_DIR.iterdir()):
        if not category_dir.is_dir():
            continue

        for pdf_path in sorted(category_dir.glob("*.pdf")):
            print(f"Loading {pdf_path} ...")
            pages = PyPDFLoader(str(pdf_path)).load()

            for page in pages:
                page.metadata["category"] = category_dir.name
                page.metadata["source"] = pdf_path.name

            documents.extend(pages)

    if not documents:
        raise SystemExit(
            "No PDFs found. Add PDFs like data/Hostel/hostel_rules.pdf, then rerun."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    print(f"Loaded {len(documents)} pages; created {len(chunks)} chunks.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"FAISS index saved to {INDEX_DIR}/")


if __name__ == "__main__":
    main()

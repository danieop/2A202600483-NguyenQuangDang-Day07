from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    OPENAI_CHAT_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    OpenAIChatLLM,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]
VINFAST_MD_CLEAN_DIR = "data/vinfast_markdown_clean"
VINFAST_MD_DIR = "data/vinfast_markdown"
VINFAST_DIR = "Vinfast"


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        print("Skipping PDF because pypdf is not installed. Run: pip install -r requirements.txt")
        return ""

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        print(f"Failed to open PDF: {path} ({exc})")
        return ""

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages).strip()


def _expand_input_paths(file_paths: list[str]) -> list[Path]:
    allowed_extensions = {".md", ".txt", ".pdf"}
    expanded: list[Path] = []
    seen: set[Path] = set()

    for raw_path in file_paths:
        path = Path(raw_path)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in allowed_extensions:
                    resolved = candidate.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        expanded.append(candidate)
            continue

        if path.is_file() and path.suffix.lower() in allowed_extensions:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                expanded.append(path)

    return expanded


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt", ".pdf"}
    documents: list[Document] = []

    for path in _expand_input_paths(file_paths):
        extension = path.suffix.lower()

        if extension not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt, .pdf)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        if extension == ".pdf":
            content = _read_pdf_text(path)
        else:
            content = path.read_text(encoding="utf-8")

        if not content.strip():
            print(f"Skipping empty content file: {path}")
            continue

        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": extension},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def chunk_documents_for_store(docs: list[Document], chunk_size: int = 1800) -> list[Document]:
    chunker = RecursiveChunker(chunk_size=chunk_size)
    chunked_docs: list[Document] = []

    for doc in docs:
        chunks = chunker.chunk(doc.content)
        if not chunks:
            continue

        if len(chunks) == 1 and chunks[0] == doc.content:
            chunked_docs.append(doc)
            continue

        for index, chunk in enumerate(chunks, start=1):
            metadata = dict(doc.metadata)
            metadata.update({"doc_id": doc.id, "chunk_index": index})
            chunked_docs.append(
                Document(
                    id=f"{doc.id}__chunk{index:03d}",
                    content=chunk,
                    metadata=metadata,
                )
            )

    return chunked_docs


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    if Path(VINFAST_MD_CLEAN_DIR).exists():
        default_inputs = [VINFAST_MD_CLEAN_DIR]
    elif Path(VINFAST_MD_DIR).exists():
        default_inputs = [VINFAST_MD_DIR]
    elif Path(VINFAST_DIR).exists():
        default_inputs = [VINFAST_DIR]
    else:
        default_inputs = SAMPLE_FILES
    files = sample_files or default_inputs
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt, .pdf")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    docs_for_store = chunk_documents_for_store(docs)
    print(f"Chunked into {len(docs_for_store)} storeable documents")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    llm_fn = demo_llm
    llm_backend_name = "demo llm fallback"
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
            llm_fn = OpenAIChatLLM(model_name=os.getenv("OPENAI_CHAT_MODEL", OPENAI_CHAT_MODEL))
            llm_backend_name = getattr(llm_fn, "_backend_name", "openai chat")
        except Exception:
            embedder = _mock_embed
            llm_fn = demo_llm
            llm_backend_name = "demo llm fallback"
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    print(f"Chat backend: {llm_backend_name}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs_for_store)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())

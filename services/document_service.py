from pathlib import Path


MAX_DOCUMENT_CONTEXT_CHARS = 12_000


def extract_document_text(path, filename):
    """Extract readable text for the chat-only document context."""
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF support requires the pypdf package.") from error
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as error:
            raise RuntimeError("DOCX support requires the python-docx package.") from error
        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    raise ValueError("Supported files: PDF, DOCX, TXT, and MD.")


def compact_document_text(text):
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) > MAX_DOCUMENT_CONTEXT_CHARS:
        return text[:MAX_DOCUMENT_CONTEXT_CHARS] + "\n\n[Document excerpt truncated for this answer.]"
    return text

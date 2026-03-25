from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import fitz


@dataclass(slots=True)
class PDFDocument:
    source_file: str
    text: str
    # FIX: Added doc_id — a stable identifier used to filter and delete all
    # chunks belonging to this PDF from the vector store. For PDFs the file
    # name is the stable key (it is also what admin_store uses for the delete
    # lookup in main.py). Previously there was no link between the admin-store
    # document record and the vector-store chunks, so deletion could only be
    # done by wiping and rebuilding the entire collection.
    doc_id: str = ""
    source_type: str = "pdf"


class PDFService:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt"}

    def __init__(self, pdf_directory: str | Path) -> None:
        self.pdf_directory = Path(pdf_directory)

    def load_documents(self) -> list[PDFDocument]:
        if not self.pdf_directory.exists():
            return []

        documents: list[PDFDocument] = []
        for file_path in sorted(self.pdf_directory.iterdir()):
            if not file_path.is_file() or file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            text = self.extract_text(file_path)
            if text:
                documents.append(
                    PDFDocument(
                        source_file=file_path.name,
                        text=text,
                        # FIX: doc_id = file name, matches document["file_name"]
                        # in admin_store so main.py can pass the right value to
                        # vector_store.delete_by_doc_id() at deletion time.
                        doc_id=file_path.name,
                        source_type=file_path.suffix.lower().lstrip("."),
                    )
                )
        return documents

    def extract_text(self, file_path: str | Path) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf_text(path)
        if suffix == ".docx":
            return self._extract_docx_text(path)
        if suffix == ".pptx":
            return self._extract_pptx_text(path)
        if suffix == ".txt":
            return self._extract_txt_text(path)
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        page_text: list[str] = []
        with fitz.open(pdf_path) as document:
            for page in document:
                page_text.append(page.get_text("text"))

        raw_text = "\n".join(page_text)
        return self._clean_text(raw_text)

    def _extract_docx_text(self, docx_path: Path) -> str:
        with ZipFile(docx_path) as archive:
            xml_bytes = archive.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        chunks = [node.text or "" for node in root.findall(".//w:t", namespace)]
        return self._clean_text(" ".join(chunks))

    def _extract_pptx_text(self, pptx_path: Path) -> str:
        with ZipFile(pptx_path) as archive:
            slide_names = sorted(
                name for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            )
            slide_text: list[str] = []
            namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            for slide_name in slide_names:
                root = ET.fromstring(archive.read(slide_name))
                texts = [node.text or "" for node in root.findall(".//a:t", namespace)]
                if texts:
                    slide_text.append(" ".join(texts))
        return self._clean_text("\n\n".join(slide_text))

    def _extract_txt_text(self, txt_path: Path) -> str:
        try:
            raw_text = txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        return self._clean_text(raw_text)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

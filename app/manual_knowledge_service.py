from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ManualKnowledgeDocument:
    source_file: str
    text: str
    # FIX: Added doc_id — the admin-store UUID for this snippet / Q&A entry.
    # Using the UUID (not the human-readable title or question) as the key
    # ensures stable matching even if the user renames the snippet or edits
    # the question text. main.py passes document["id"] to
    # vector_store.delete_by_doc_id() at deletion time.
    doc_id: str = ""


@dataclass(slots=True)
class TextSnippetRecord:
    id: str
    title: str
    content: str


@dataclass(slots=True)
class QARecord:
    id: str
    question: str
    answer: str


class ManualKnowledgeService:
    def __init__(self, snippets_directory: str | Path, qa_directory: str | Path) -> None:
        self.snippets_directory = Path(snippets_directory)
        self.qa_directory = Path(qa_directory)

    def load_documents(self) -> list[ManualKnowledgeDocument]:
        documents: list[ManualKnowledgeDocument] = []

        for record in self.list_text_snippets().values():
            content = record.content.strip()
            if not content:
                continue
            documents.append(
                ManualKnowledgeDocument(
                    source_file=f"Snippet: {record.title}",
                    text=f"Title: {record.title}\n\nSnippet:\n{content}",
                    # FIX: Use the record UUID, not the title, so the doc_id
                    # remains stable across renames.
                    doc_id=record.id,
                )
            )

        for record in self.list_qa().values():
            question = record.question.strip()
            answer = record.answer.strip()
            if not question or not answer:
                continue
            documents.append(
                ManualKnowledgeDocument(
                    source_file=f"Q&A: {question}",
                    text=f"Question: {question}\nAnswer: {answer}",
                    # FIX: Same — use the record UUID.
                    doc_id=record.id,
                )
            )

        return documents

    def list_text_snippets(self) -> dict[str, TextSnippetRecord]:
        return self._load_records(self.snippets_directory, "title", "content", TextSnippetRecord)

    def list_qa(self) -> dict[str, QARecord]:
        return self._load_records(self.qa_directory, "question", "answer", QARecord)

    def save_text_snippet(self, record_id: str, title: str, content: str) -> TextSnippetRecord:
        self.snippets_directory.mkdir(parents=True, exist_ok=True)
        payload = {"id": record_id, "title": title.strip(), "content": content.strip()}
        (self.snippets_directory / f"{record_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return TextSnippetRecord(**payload)

    def save_qa(self, record_id: str, question: str, answer: str) -> QARecord:
        self.qa_directory.mkdir(parents=True, exist_ok=True)
        payload = {"id": record_id, "question": question.strip(), "answer": answer.strip()}
        (self.qa_directory / f"{record_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return QARecord(**payload)

    def delete_text_snippet(self, record_id: str) -> None:
        (self.snippets_directory / f"{record_id}.json").unlink(missing_ok=True)

    def delete_qa(self, record_id: str) -> None:
        (self.qa_directory / f"{record_id}.json").unlink(missing_ok=True)

    @staticmethod
    def _load_records(directory: Path, first_key: str, second_key: str, model):
        if not directory.exists():
            return {}

        records: dict[str, object] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            record_id = str(payload.get("id", "")).strip()
            first = str(payload.get(first_key, "")).strip()
            second = str(payload.get(second_key, "")).strip()
            if not record_id or not first or not second:
                continue
            records[record_id] = model(**{"id": record_id, first_key: first, second_key: second})

        return records
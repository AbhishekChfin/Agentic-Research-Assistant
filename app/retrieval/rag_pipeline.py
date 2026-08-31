"""
No layoutparser
No detectron2
Still extract:
- page text
- tables
- table id/title from nearby text
- table rows in sentence format
- relative source_url
"""
import re
from pathlib import Path
from typing import Any
import pdfplumber
from app.core.schemas import EvidenceItem
from app.retrieval.embedder import BGEEmbedder
from app.retrieval.vector_index import VectorIndex

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    /"data"
    /"processed"
    /"embeddings"
    /"financial_machine_learning_sentence_700_overlap_120_embedded.jsonl"
)


class RAGSearch:
    """
    Args:
        embeddings_path: JSONL file containing embedded chunk records.
        top_k: Default number of matching chunks to return.

    Returns:
        RAGSearch instance that lazily builds and reuses a vector index.
    """

    def __init__(
        self,
        embeddings_path: Path = EMBEDDINGS_PATH,
        top_k: int = 10,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self.embeddings_path = embeddings_path
        self.top_k = top_k
        self._vector_index: VectorIndex | None = None

    def _get_vector_index(self) -> VectorIndex:
        if self._vector_index is None:
            vector_index = VectorIndex(embedder=BGEEmbedder())
            records = vector_index.load_embeddings(self.embeddings_path)
            vector_index.build(records)
            self._vector_index = vector_index

        return self._vector_index

    def search(self, query: str, top_k: int | None = None) -> list[EvidenceItem]:
        result_limit = self.top_k if top_k is None else top_k
        if result_limit <= 0:
            raise ValueError("top_k must be greater than 0")

        results = self._get_vector_index().search(query, top_k=result_limit)

        return [
            EvidenceItem(
                claim=result["text"],
                source_title=result["source_title"],
                source_url=result["source_url"],
                confidence=result["similarity_score"],
            )
            for result in results
        ]


class PDFExtractor:
    def __init__(self, pdf_path: str, page_offset: int = 0):
        self.pdf_path = Path(pdf_path)
        source_path = self.pdf_path
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path

        try:
            self.source_url = source_path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            self.source_url = self.pdf_path.as_posix()

        self.page_offset = page_offset

    def extract(self) -> list[dict[str, Any]]:
        records = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                display_page = page_number - self.page_offset

                page_text = page.extract_text(
                    x_tolerance=1,
                    y_tolerance=3,
                    layout=True,
                ) or ""

                clean_page_text = self._clean(page_text)

                if clean_page_text:
                    records.append(
                        {
                            "text": clean_page_text,
                            "source_title": self.pdf_path.name,
                            "source_url": self.source_url,
                            "pdf_page": page_number,
                            "page": display_page,
                            "content_type": "text",
                        }
                    )

                tables = page.extract_tables()

                for table_index, table in enumerate(tables, start=1):
                    table_title = self._find_table_title(page_text, page_number, table_index)

                    if not table or len(table) < 2:
                        continue

                    headers = [self._clean(cell) for cell in table[0]]
                    rows = table[1:]

                    for row_index, row in enumerate(rows, start=1):
                        values = [self._clean(cell) for cell in row]
                        pairs = []

                        for header, value in zip(headers, values):
                            if header and value:
                                pairs.append(f"{header}: {value}")

                        if not pairs:
                            continue

                        records.append(
                            {
                                "text": f"{table_title}. " + ". ".join(pairs) + ".",
                                "source_title": self.pdf_path.name,
                                "source_url": self.source_url,
                                "pdf_page": page_number,
                                "page": display_page,
                                "content_type": "table_row",
                                "table_id": table_title,
                                "row_id": row_index,
                            }
                        )
        return records

    def _find_table_title(self, page_text: str, page_number: int, table_index: int) -> str:
        lines = [self._clean(line) for line in page_text.splitlines() if self._clean(line)]

        for i, line in enumerate(lines):
            match = re.search(r"Table\s+\d+(\.\d+)*", line, re.IGNORECASE)
            if match:
                table_id = match.group(0)
                caption = lines[i + 1] if i + 1 < len(lines) else ""

                if caption:
                    return f"{table_id}: {caption}"

                return table_id

        return f"Page {page_number} Table {table_index}"

    def _clean(self, value: Any) -> str:
        if value is None:
            return ""

        text = str(value)

        # fix hyphenated line breaks: "com-\nbination" -> "combination"
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # normalize spaces around newlines
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)

        # collapse 3+ newlines into one paragraph break
        text = re.sub(r"\n{3,}", "\n\n", text)

        # normalize repeated spaces/tabs inside lines
        text = re.sub(r"[ \t]+", " ", text)

        # remove spaces before punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)

        # optional PDF symbol fixes if you already use this method
        text = self._fix_pdf_symbols(text)

        return text.strip()
    
    def _clean_for_embedding(self, value: Any) -> str:
        text = self._clean(value)

        # turn all whitespace, including paragraph breaks, into single spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()
    
    def _fix_pdf_symbols(self, text: str) -> str:
        replacements = {
            "(cid:15)": "ϵ",
            "(cid:12)": "β",
            "(cid:11)": "α",
            "(cid:22)": "μ",
            "(cid:27)": "π",
        }

        for bad, good in replacements.items():
            text = text.replace(bad, good)

        return text



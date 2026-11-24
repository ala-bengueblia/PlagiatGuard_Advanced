from pathlib import Path
import csv
import io
import json

from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document


TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".java",
    ".cpp", ".c", ".h", ".html", ".htm", ".xml",
    ".json", ".csv", ".tsv", ".ipynb"
}


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if ext == ".docx":
        doc = Document(io.BytesIO(data))
        parts = [paragraph.text for paragraph in doc.paragraphs]

        for table in doc.tables:
            for row in table.rows:
                parts.append(
                    " | ".join(cell.text for cell in row.cells)
                )

        return "\n".join(parts)

    if ext == ".odt":
        from odf.opendocument import load
        from odf import text as odf_text

        doc = load(io.BytesIO(data))
        return "\n".join(
            str(node)
            for node in doc.getElementsByType(odf_text.P)
        )

    if ext not in TEXT_EXTENSIONS:
        raise ValueError(f"Format non pris en charge : {ext}")

    decoded = data.decode("utf-8", errors="replace")

    if ext in {".html", ".htm", ".xml"}:
        return BeautifulSoup(
            decoded, "html.parser"
        ).get_text(" ", strip=True)

    if ext == ".ipynb":
        notebook = json.loads(decoded)
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )

    if ext in {".csv", ".tsv"}:
        delimiter = "\t" if ext == ".tsv" else ","
        rows = csv.reader(
            io.StringIO(decoded), delimiter=delimiter
        )
        return "\n".join(" ".join(row) for row in rows)

    return decoded
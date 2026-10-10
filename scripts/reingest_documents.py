"""Re-ingest documents from the Django database into the vector index.

Run from the project root with the server stopped:
    python scripts/reingest_documents.py          # all documents
    python scripts/reingest_documents.py 5 7      # only these document IDs
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.config.settings")

import django
django.setup()

from api.models import Document
from docurag import pipeline


def main(ids):
    docs = Document.objects.all().order_by("id")
    if ids:
        docs = docs.filter(id__in=ids)

    for doc in docs:
        path = doc.file.path
        if not os.path.exists(path):
            print(f"[skip] {doc.id} {doc.filename}: file missing at {path}")
            continue
        print(f"Re-ingesting {doc.filename} (ID {doc.id})")
        try:
            result = pipeline.ingest_file(path, document_id=str(doc.id))
            total = result.get("chunks_indexed", 0)
            doc.pages = result.get("pages")
            doc.chunk_counts = {"total": total}
            if total == 0:
                doc.status, doc.error = "failed", "No chunks were produced"
            else:
                doc.status, doc.error = "ready", None
        except Exception as exc:
            doc.status, doc.error = "failed", str(exc)
            print(f"  failed: {exc}")
        doc.save()
        print(f"  -> {doc.status}, chunks: {doc.chunk_counts.get('total')}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])

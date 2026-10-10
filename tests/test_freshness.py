import os
import shutil
import pytest
from pathlib import Path
from docurag.pipeline import ingest_file, ask
from docurag.config import get_settings

def test_mode_switch_cache_poisoning(tmp_path, monkeypatch):
    # Setup isolated storage
    storage_dir = tmp_path / "storage"
    monkeypatch.setenv("DOCURAG_STORAGE", str(storage_dir))

    # Create a dummy PDF file
    pdf_path = tmp_path / "test.pdf"
    try:
        import fitz
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()
    except ImportError:
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n111\n%%EOF\n")

    # 1. Ingest in FAKE mode
    monkeypatch.setenv("DOCURAG_FAKES", "1")
    settings = get_settings(reload=True)

    report_fake = ingest_file(str(pdf_path))
    assert report_fake["status"] == "ready"

    # Verify it's indexed (in fake mode, it might just return fake results)
    res_fake = ask("What is in the document?")
    # In fake mode, the answer is deterministic based on the fake generator
    fake_answer = res_fake.answer

    # 2. Switch to LIVE mode (FAKES=0)
    monkeypatch.setenv("DOCURAG_FAKES", "0")
    # Force reload settings
    settings = get_settings(reload=True)

    # The ChunkStore is typically initialized when the pipeline is first used.
    # We need to trigger the store initialization to run _check_and_wipe_stale_index.
    # In the current architecture, the store is often a singleton or part of the pipeline.
    # For the test, we can manually trigger the store's logic if needed,
    # but the requirement is that the system handles it.

    # Ingest again in LIVE mode
    # This should trigger the wipe of storage/index and storage/cache
    report_live = ingest_file(str(pdf_path))

    # Check if cache was wiped (the fake cache files should be gone)
    # Note: disk_cache uses settings.cache_dir / namespace
    cache_dir = settings.cache_dir
    fake_cache_files = list(cache_dir.glob("**/*.json"))
    # Since we switched to live, any remaining files would be from the first run
    # if they weren't wiped.
    assert len(fake_cache_files) == 0, "Cache should have been wiped when switching from FAKE to LIVE"

    # Verify that the result is no longer the 'fake' one
    res_live = ask("What is in the document?")
    # If it's live, it should either fail (because we have no real LLM key)
    # or return something different from the hardcoded fake result.
    # But the key is that it doesn't return the cached FAKE result.
    assert res_live.answer != fake_answer, "Should not return cached fake result in live mode"

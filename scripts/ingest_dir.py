import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure the project root is in the Python path
# This allows the script to find the 'docurag' package regardless of where it's run from
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from docurag.pipeline import ingest_file, ingest_directory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="DocuRAG Bulk Ingestion Tool")
    parser.add_argument("path", type=str, help="Path to the document or directory to ingest")
    parser.add_argument("--collection", type=str, default="default", help="Collection ID (default: default)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        path = Path(args.path).resolve()
    except Exception as e:
        logger.error(f"Invalid path provided: {args.path}. Error: {e}")
        sys.exit(1)

    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    if path.is_dir():
        logger.info(f"Ingesting directory: {path}")
        try:
            report = ingest_directory(str(path), collection_id=args.collection)
            logger.info(f"Successfully ingested directory. Report: {report}")
        except Exception as e:
            logger.exception(f"Failed to ingest directory {path}: {e}")
            sys.exit(1)
    else:
        logger.info(f"Ingesting file: {path}")
        try:
            report = ingest_file(str(path), collection_id=args.collection)
            logger.info(f"Successfully ingested file. Report: {report}")
        except Exception as e:
            logger.exception(f"Failed to ingest file {path}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()

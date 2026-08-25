import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import ingest_folder

if __name__ == "__main__":
    result = ingest_folder()
    print(f"Ingestion completed — documents: {result['documents']}, chunks: {result['chunks']}")

import sys
from pathlib import Path
from src.ingestion.partitioner import DocumentPartitioner


def test_partitioner(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"File {file_path} does not exist.")
        return

    partitioner = DocumentPartitioner()
    chunks = partitioner.partition_and_chunk(path)

    print(f"Number of chunks produced: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:")
        print(f"  Type: {type(chunk)}")
        print(f"  Text: {chunk.text[:100]}...")
        print(f"  Metadata: {chunk.metadata.to_dict()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_partitioner.py <file_path>")
    else:
        test_partitioner(sys.argv[1])

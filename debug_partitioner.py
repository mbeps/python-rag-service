import sys
from pathlib import Path
from src.ingestion.partitioner import DocumentPartitioner


def run_partitioner_debug(file_path: str) -> None:
    """Debug utility to test document partitioner on a single file.

    Args:
        file_path: Path to the file to partition
    """
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
        print("Usage: python debug_partitioner.py <file_path>")
    else:
        run_partitioner_debug(sys.argv[1])

from pathlib import Path
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element


class DocumentPartitioner:
    """
    Handles partitioning and chunking of documents using logical layout awareness.
    Uses 'unstructured' library to extract elements and group them by titles.
    """

    def partition_and_chunk(self, file_path: Path) -> List[Element]:
        """
        Partitions the document into elements and applies title-based chunking.

        Args:
            file_path (Path): Path to the document file to be processed.

        Returns:
            List[Element]: A list of chunked elements from the document.
        """
        # Partition the document
        # Note: partition_pdf is heavy. In production, consider offloading to a worker.
        elements = partition_pdf(
            filename=str(file_path), pdf_infer_table_structure=True, strategy="hi_res"
        )

        # Chunk elements by title to preserve semantic structure
        chunks = chunk_by_title(
            elements,
            combine_text_under_n_chars=200,
            max_characters=1500,
            multipage_sections=True,
        )

        return chunks

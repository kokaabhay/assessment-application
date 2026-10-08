import re
import hashlib
from config import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks(
    text: str,
    document_id: str,
    source_file: str,
):
    paragraphs = re.split(r"\n\s*\n", text.strip())

    chunks = []
    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If adding the paragraph stays within the limit
        if len(current) + len(paragraph) <= CHUNK_SIZE:
            current = f"{current}\n\n{paragraph}" if current else paragraph

        else:
            # Save current chunk
            if current:
                chunks.append(current)

            # Handle a paragraph that is itself too large
            if len(paragraph) > CHUNK_SIZE:
                start = 0

                while start < len(paragraph):

                    end = start + CHUNK_SIZE

                    chunk = paragraph[start:end]

                    chunks.append(chunk)

                    start = end - CHUNK_OVERLAP

                current = ""

            else:
                # Start next chunk with overlap
                overlap = current[-CHUNK_OVERLAP:] if current else ""

                current = f"{overlap}\n\n{paragraph}" if overlap else paragraph

    if current:
        chunks.append(current)

    result = []

    for index, content in enumerate(chunks):

        # Stable ID based on document + position
        chunk_id = f"{document_id}_chunk_{index:04d}"

        result.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source_file": source_file,
                "chunk_index": index,
                "content": content,
                "document_type": "HR",
                "category": "HR Policy",
            }
        )

    return result

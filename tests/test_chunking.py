from ragforge.chunking import chunk_documents
from ragforge.schemas import Document


def test_chunking_preserves_source():
    doc = Document("First sentence. Second sentence. Third sentence.", "demo.txt")
    chunks = chunk_documents([doc])
    assert chunks
    assert all(c.source == "demo.txt" for c in chunks)
    assert "First sentence" in chunks[0].text

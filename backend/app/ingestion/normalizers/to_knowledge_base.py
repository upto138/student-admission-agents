"""
Normalizer: put parsed data into the knowledge base (Azure or Chroma).
Called after the crawler has parsed HTML into structured text.
"""
from app.tools.rag_tool import add_to_knowledge_base

def ingest_university_data(parsed_items: list[dict]) -> list[dict]:
    """
    Args:
        parsed_items: List[{content, source_url, university, year}]

    Returns:
        List of ingestion results
    """
    results = []
    for item in parsed_items:
        result = add_to_knowledge_base(
            content=item["content"],
            source_url=item.get("source_url", ""),
            university=item.get("university", ""),
            year=item.get("year", "2025"),
        )
        results.append(result)
    return results
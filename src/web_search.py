"""
Web search tool for agentic RAG fallback.

Provides DuckDuckGoSearchTool as a last-resort retrieval source when local
FAISS/BM25 retrieval produces poor results. Results are formatted as
document dicts compatible with the RAG pipeline.
"""

from typing import Any, Dict, List

from loguru import logger


class DuckDuckGoSearchTool:
    """Web search via DuckDuckGo (free, no API key required).

    Lazy-imports ddgs to avoid loading it when web search
    is not enabled. Returns results as doc-like dicts with 'content'
    and 'source' keys for pipeline compatibility.

    Attributes:
        max_results: Maximum number of search results to return.
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search DuckDuckGo and return formatted results.

        Args:
            query: Search query string.

        Returns:
            List of document dicts with 'content' and 'source' keys.
            Returns empty list on error (graceful degradation).
        """
        try:
            from ddgs import DDGS

            results = DDGS().text(query, max_results=self.max_results)

            docs = []
            for r in results:
                docs.append(
                    {
                        "content": f"{r['title']}\n{r['body']}",
                        "source": r.get("href", "web_search"),
                    }
                )

            logger.info(f"DuckDuckGo returned {len(docs)} results for: '{query}'")
            return docs

        except Exception as e:
            logger.warning(f"Web search failed: {e}. Continuing with local docs only.")
            return []

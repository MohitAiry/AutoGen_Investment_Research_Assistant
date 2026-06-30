import warnings
from duckduckgo_search import DDGS

warnings.filterwarnings("ignore")
def search_news(query: str) -> str:
    """Search the web for real-time news about a company."""
    print(f"\n[TOOL CALL] Searching the live web for: {query}...\n")
    try:
        # We use .news() instead of .text() to get actual news headlines and facts, not generic SEO landing pages.
        results = DDGS().news(query, max_results=5)
        if not results:
            return "No recent news found on the web."
        
        summary = ""
        for r in results:
            # News results usually have 'title', 'body', 'date', and 'source'
            date = r.get('date', '')[:10]
            summary += f"- [{date} | {r.get('source', 'News')}] {r.get('title', '')}: {r.get('body', '')}\n"
        return summary
    except Exception as e:
        return f"Error performing live web search: {e}"

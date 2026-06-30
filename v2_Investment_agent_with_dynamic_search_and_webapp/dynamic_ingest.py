import os
import yfinance as yf
import chromadb
from chromadb.utils import embedding_functions

def fetch_financial_data(ticker_symbol):
    """Fetches live financial data from Yahoo Finance and formats it as text."""
    print(f"Fetching live data for {ticker_symbol} from Yahoo Finance...")
    ticker = yf.Ticker(ticker_symbol)
    
    # Get basic info (ratios, margins, etc.)
    info = ticker.info
    
    # Create a structured text document from the data
    doc_text = f"--- Document: ratios (Live Data) ---\n"
    doc_text += f"{info.get('longName', ticker_symbol)} Financial Ratios & Balance Sheet Summary\n"
    doc_text += f"Sector: {info.get('sector', 'N/A')}, Industry: {info.get('industry', 'N/A')}\n\n"
    
    doc_text += "1. Valuation Metrics:\n"
    doc_text += f"- Price-to-Earnings (P/E) Ratio: {info.get('trailingPE', 'N/A')}\n"
    doc_text += f"- Price-to-Sales (P/S) Ratio: {info.get('priceToSalesTrailing12Months', 'N/A')}\n"
    doc_text += f"- Forward P/E: {info.get('forwardPE', 'N/A')}\n\n"
    
    doc_text += "2. Profitability and Return Ratios:\n"
    doc_text += f"- Return on Equity (ROE): {info.get('returnOnEquity', 'N/A')}\n"
    doc_text += f"- Gross Margin: {info.get('grossMargins', 'N/A')}\n"
    doc_text += f"- Operating Margin: {info.get('operatingMargins', 'N/A')}\n"
    doc_text += f"- Profit Margin: {info.get('profitMargins', 'N/A')}\n\n"
    
    doc_text += "3. Liquidity and Solvency:\n"
    doc_text += f"- Current Ratio: {info.get('currentRatio', 'N/A')}\n"
    doc_text += f"- Total Debt: {info.get('totalDebt', 'N/A')}\n"
    doc_text += f"- Cash and Cash Equivalents: {info.get('totalCash', 'N/A')}\n\n"
    
    doc_text += "4. Growth Metrics:\n"
    doc_text += f"- Revenue Growth: {info.get('revenueGrowth', 'N/A')}\n"
    doc_text += f"- Earnings Growth: {info.get('earningsGrowth', 'N/A')}\n"
    
    # Get recent news directly from yfinance
    news = ticker.news
    if news:
        doc_text += "\n--- Document: recent_news (Live Data) ---\n"
        for item in news[:3]:
            doc_text += f"Title: {item.get('title', 'N/A')}\n"
            doc_text += f"Publisher: {item.get('publisher', 'N/A')}\n\n"
            
    return doc_text, info.get('longName', ticker_symbol).lower().replace(" ", "_")

def ingest_single_ticker(ticker_symbol, year="2024"):
    """JIT (Just-In-Time) ingestion for a single ticker to keep ChromaDB updated."""
    try:
        content, company_name = fetch_financial_data(ticker_symbol)
        
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text:latest"
        )
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(name="financial_docs", embedding_function=ollama_ef)
        
        metadata = {
            "company": ticker_symbol.lower(),
            "doc_type": "yfinance_live_data",
            "year": str(year),
            "filename": f"{ticker_symbol.lower()}_live_data.txt"
        }
        
        print(f"\n[JIT INGESTION] Pushing live data for {ticker_symbol} into ChromaDB...")
        collection.upsert(
            documents=[content],
            metadatas=[metadata],
            ids=[f"doc_{ticker_symbol.lower()}_live_data"]
        )
        return True
    except Exception as e:
        print(f"[JIT INGESTION] Failed to fetch or ingest data: {e}")
        return False

def main():
    print("Starting DYNAMIC document ingestion pipeline...")
    
    try:
        print("Connecting to local Ollama embedding model...")
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text:latest"
        )
    except Exception as e:
        print(f"Error loading Ollama embedding function: {e}")
        return

    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(
            name="financial_docs", 
            embedding_function=ollama_ef
        )
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return

    # List of tickers to dynamically ingest
    tickers = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "GOOGL"]
    
    for ticker in tickers:
        ingest_single_ticker(ticker, "2024")
        
    print(f"Total documents in the 'financial_docs' collection: {collection.count()}")

if __name__ == "__main__":
    main()

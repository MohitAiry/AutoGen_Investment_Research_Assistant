import chromadb
from chromadb.utils import embedding_functions

def retrieve_financial_context(company: str, year: str = None) -> str:
    """
    Queries ChromaDB with the company name (and optional year) and financial sub-queries to retrieve context.
    Returns a formatted string of the top-k relevant document chunks.
    """
    try:
        # Load the Ollama embedding function used in ingestion
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text:latest"
        )
        
        # Connect to the persistent ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get the collection
        collection = client.get_collection(
            name="financial_docs",
            embedding_function=ollama_ef
        )
    except Exception as e:
        print(f"Error accessing ChromaDB: {e}")
        return "Error: Could not retrieve context due to database access issues."
        
    # Define sub-queries to cover various financial domains for the specific company
    queries = [
        f"{company} revenue growth earnings",
        f"{company} financial ratios debt margins",
        f"{company} risk factors headwinds"
    ]
    
    retrieved_chunks = []
    seen_docs = set()
    
    try:
        # Perform similarity search for each query
        for query in queries:
            # Construct the metadata filter
            where_clause = {"company": company.lower()}
            if year:
                where_clause = {
                    "$and": [
                        {"company": company.lower()},
                        {"year": str(year)}
                    ]
                }
                
            results = collection.query(
                query_texts=[query],
                n_results=3,  # Get top 3 results per query
                # We can optionally filter by metadata if we only want documents for this company
                where=where_clause
            )
            
            # Process the results
            if results and 'documents' in results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    doc_id = results['ids'][0][i]
                    metadata = results['metadatas'][0][i]
                    
                    # Deduplicate chunks based on document ID
                    if doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        year = metadata.get('year', 'Unknown Year')
                        doc_type = metadata.get('doc_type', 'Unknown Doc Type')
                        
                        formatted_chunk = f"--- Document: {doc_type} ({year}) ---\n{doc}\n"
                        retrieved_chunks.append(formatted_chunk)
                        
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return f"Error: Failed to execute queries for {company}."
        
    if not retrieved_chunks:
        return f"No financial documents found in the database for company: {company}."
        
    # Combine all chunks into a single context string
    full_context = "\n".join(retrieved_chunks)
    return full_context

if __name__ == "__main__":
    # Test the retriever locally
    print("Testing retriever for Apple...")
    print(retrieve_financial_context("Apple"))

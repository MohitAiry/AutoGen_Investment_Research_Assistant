import os
import glob
import chromadb
from chromadb.utils import embedding_functions

def main():
    print("Starting document ingestion pipeline...")
    
    # 1. Load the all-MiniLM-L6-v2 embedding function
    # We use a try-except block here because downloading the model can sometimes fail due to network issues.
    try:
        print("Connecting to local Ollama embedding model...")
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text:latest"
        )
    except Exception as e:
        print(f"Error loading Ollama embedding function: {e}")
        print("Tip: Make sure Ollama is running and 'nomic-embed-text:latest' is pulled.")
        return

    # 2. Create or load the financial_docs ChromaDB collection
    try:
        # We use a persistent client so the data is saved to the disk in the ./chroma_db folder.
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # get_or_create_collection ensures we don't crash if the collection already exists.
        collection = client.get_or_create_collection(
            name="financial_docs", 
            embedding_function=ollama_ef
        )
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return

    # 3. Read documents from the /data/financial/ directory
    data_dir = "./data/financial/"
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Please create it and add text documents.")
        return
        
    filepaths = glob.glob(os.path.join(data_dir, "*.txt"))
    
    if not filepaths:
        print(f"No text files found in {data_dir}. Please add some financial documents.")
        return
        
    print(f"Found {len(filepaths)} documents to ingest.")

    documents = []
    metadatas = []
    ids = []

    # Iterate through the files and extract text and metadata
    for i, filepath in enumerate(filepaths):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract metadata from filename convention: {company}_{doc_type}_{year}.txt
            # Example: apple_earnings_transcript_2023.txt
            filename = os.path.basename(filepath)
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('_')
            
            # Simple heuristic for metadata extraction based on our filename convention
            if len(parts) >= 3:
                company = parts[0].lower()
                year = parts[-1]
                doc_type = "_".join(parts[1:-1])
            else:
                company = "unknown"
                year = "unknown"
                doc_type = "unknown"
                
            metadata = {
                "company": company,
                "doc_type": doc_type,
                "year": year,
                "filename": filename
            }
            
            documents.append(content)
            metadatas.append(metadata)
            # Create a unique ID for each document chunk. For simplicity, we use the filename.
            ids.append(f"doc_{filename}")
            
            print(f"Loaded: {filename} (Company: {company})")
            
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")

    # 4. Add the documents to the ChromaDB collection
    if documents:
        try:
            print("Adding documents to ChromaDB...")
            # We use upsert so that if a document with the same ID already exists, it is updated
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print("Ingestion complete!")
            print(f"Total documents in the 'financial_docs' collection: {collection.count()}")
        except Exception as e:
            print(f"Error adding documents to collection: {e}")

if __name__ == "__main__":
    main()

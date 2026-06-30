# AutoGen Investment Research Assistant (Lab 1)

This project demonstrates an advanced multi-agent system built using Microsoft AutoGen, Ollama, and ChromaDB. It features a robust Retrieval-Augmented Generation (RAG) architecture where a `ResearchAgent` drafts financial investment briefs based on precise data retrieved from ChromaDB, and a `CriticAgent` critically reviews and enforces the quality of the brief before approval.

## Table of Contents
1. [Assignment Overview: What Are We Building?](#1-assignment-overview-what-are-we-building)
2. [Chapter 1: Conceptual and Methodological Foundations](#2-chapter-1-conceptual-and-methodological-foundations)
3. [Why Use AutoGen? (Comparison to Other Frameworks)](#3-why-use-autogen-comparison-to-other-frameworks)
4. [Project Setup & Prerequisites](#4-project-setup--prerequisites)
5. [How to Run](#5-how-to-run)
6. [The Full Pipeline Explained](#6-the-full-pipeline-explained)
7. [In-Depth Python Scripts and Function Definitions](#7-in-depth-python-scripts-and-function-definitions)

---

## 1. Assignment Overview: What Are We Building?

Junior analysts at investment banks spend hours reading earnings call transcripts, financial ratios, and market summaries to produce a single investment brief. This manual synthesis is slow, inconsistent across analysts, and prone to confirmation bias.

In this assignment, you are building an **automated, multi-agent AI investment research assistant**. The goal is to create a system that can accurately and autonomously synthesize financial data into professional briefs. 

We are building a **two-agent AutoGen system**:
- **The Research Agent (The Analyst):** Retrieves exact financial context from a local knowledge base (vector database) and drafts an initial investment brief.
- **The Critic Agent (The Senior Manager):** Reviews the draft against a strict checklist (checking for concrete numbers, risk factors, and minimum word counts). It provides harsh, constructive feedback to the Research Agent.

This system iteratively improves the document through autonomous conversation until the Senior Manager approves it, after which it is saved to your local disk.

---

## 2. Chapter 1: Conceptual and Methodological Foundations

To truly understand this assignment, we must look beyond the code and examine the theoretical and methodological paradigms that power this application. This section acts as a comprehensive primer on the underlying science of multi-agent systems and retrieval-based AI.

### 2.1 The Paradigm Shift: From Monolithic LLMs to Multi-Agent Systems
Historically, interacting with Large Language Models (LLMs) involved a "zero-shot" or "few-shot" prompting methodology: a user sends a single query, and the LLM returns a single block of text. While powerful, this monolithic approach suffers from severe limitations:
- **Lack of Self-Correction:** A single LLM pass has no built-in mechanism to step back, review its own work, and iteratively improve it.
- **Context Dilution:** If you ask a single agent to "be a creative writer AND a strict fact-checker," the model's attention is divided, often leading to compromises in both areas.

**The Multi-Agent Methodology:** Frameworks like Microsoft AutoGen introduce a profound shift. Instead of one massive prompt, we enforce a **Cognitive Division of Labor**. By spawning multiple distinct agents (instances of an LLM), we can simulate complex organizational structures. In this lab, we map a human organizational workflow—the Junior Analyst and the Senior Manager—directly onto AI agents. This allows each agent to operate with a highly focused, singular objective, dramatically increasing the logical rigor of the final output.

### 2.2 The Epistemic Anchor: Retrieval-Augmented Generation (RAG)
LLMs possess vast latent knowledge, but they are frozen in time (the "Knowledge Cutoff" problem) and prone to confidently asserting falsehoods (Hallucinations). In financial research, hallucinating a revenue figure or a P/E ratio is catastrophic.

To solve this, our methodology employs **Retrieval-Augmented Generation (RAG)**. RAG explicitly separates *knowledge* from *reasoning*:
1. **The Embedding Phase (Vectorization):** When we ingest financial documents into ChromaDB, we are not just storing text. We use an embedding model (`nomic-embed-text`) to translate human language into high-dimensional mathematical vectors. 
2. **The Retrieval Phase (Semantic Search):** When the system searches for "Apple risk factors," it translates that query into a vector and uses **Cosine Similarity** to find the closest matching document vectors in the database. This allows the system to find relevant paragraphs based on semantic meaning, even if the exact keywords don't match.
3. **The Generation Phase:** The retrieved facts act as an "epistemic anchor." We inject these hard facts into the `ResearchAgent`'s context window, forcing the LLM to reason *only* over the provided data rather than relying on its unreliable internal memory.

### 2.3 The AutoGen Methodology: Adversarial Review Loops
The core methodology of this lab is the **Adversarial Review Loop**. In AutoGen, agents do not simply pass data down a static pipeline; they engage in turn-based dialogue, reacting dynamically to one another.

- **The Synthesizer (`ResearchAgent`):** Operates with a creative/analytical mandate. Its goal is to weave the raw retrieved data into a cohesive, professional narrative.
- **The Gatekeeper (`CriticAgent`):** Operates with an adversarial, strictly logical mandate. It evaluates the synthesizer's output against a rigid programmatic checklist. 

This creates a game-theoretic dynamic. The ResearchAgent tries to complete the assignment, and the CriticAgent actively tries to find flaws in it. The CriticAgent will systematically reject the text if it spots vague qualitative language (e.g., "margins were good") instead of the required quantitative data (e.g., "margins were 44.1%"). This continuous feedback loop forces the LLM to self-correct, simulating hours of human drafting and revising in mere seconds.

### 2.4 State Management and Deterministic Bounds
A major methodological challenge in autonomous agent systems is the risk of infinite loops—what happens if the Critic and the Analyst can never agree? 
AutoGen solves this through deterministic state management constraints. We implement a `max_consecutive_auto_reply` threshold. The loop operates under the following logic:
- **Success State:** If the Critic is satisfied, it outputs the exact string `"APPROVED"`. A custom callback intercepts this string and halts the loop triumphantly.
- **Fallback State:** If the agents iterate 3 times without reaching an agreement, the system forcefully terminates the loop. This ensures computation bounds are respected and API/hardware resources are not exhausted indefinitely.

---

## 3. Why Use AutoGen? (Comparison to Other Frameworks)

When building LLM applications, developers often choose between various orchestration frameworks. Here is why AutoGen is uniquely suited for complex, iterative tasks compared to other popular tools like LangChain or LlamaIndex:

- **Conversational Design vs. Static Chains:** Frameworks like LangChain traditionally emphasize directed acyclic graphs (DAGs) and rigid execution chains (e.g., Prompt A → Tool B → Prompt C). AutoGen, however, is fundamentally built on **autonomous dialogue**. Agents are independent entities that converse dynamically. If something goes wrong, the agents can naturally discuss and fix it without needing a hardcoded fallback chain.
- **Multi-Agent First:** While other frameworks have retrofitted multi-agent capabilities, AutoGen was designed from the ground up to support complex, multi-agent topologies (e.g., group chats with 5+ agents, hierarchical agent setups, joint task forces).
- **Reduced Hallucinations:** Because AutoGen relies on a reviewer/critic pattern, the system is inherently resistant to hallucinations. An LLM might confidently hallucinate in a single prompt, but in AutoGen, the `CriticAgent` acts as an independent adversarial filter to verify facts before the final output is accepted.

---

## 4. Project Setup & Prerequisites

1. **Conda Environment:** Make sure you have Anaconda or Miniconda installed. Activate the environment:
   ```bash
   conda activate agentic
   ```

2. **Install Dependencies:**
   Install the required packages using pip. We use the `ag2` package (the new name for the autogen library) along with the `openai` extra, as the OpenAI client is used to talk to our local Ollama server:
   ```bash
   pip install autogen pyautogen 'ag2[openai]' chromadb python-dotenv
   ```

3. **Ollama:** You need Ollama installed and running locally with the `llama3.1:latest` model as well as the `nomic-embed-text:latest` embedding model. Start your local server:
   ```bash
   ollama serve
   ```
   *Note: Ensure you have pulled the models by running `ollama pull llama3.1:latest` and `ollama pull nomic-embed-text:latest` in a separate terminal before running the scripts.*

4. **Prepare Data:**
   The `data/financial/` directory contains highly realistic, unstructured text files representing 5 different tech companies (Apple, Nvidia, Microsoft, Tesla, Amazon) across two fiscal years (2023 and 2024). For every company and year, there are two distinct types of data files you will interact with:
   * **The `_earnings_` Transcript Files:** These files mimic the literal spoken words of executives (CEOs/CFOs) and Wall Street analysts during a live quarterly earnings call. They are highly qualitative, narrative-driven, and contain forward-looking macroeconomic outlooks, strategic visions, and risk factor disclosures (e.g., EU regulations, supply chain constraints).
   * **The `_ratios_` Quantitative Files:** These files mimic hard, numerical data extracted from a company's SEC 10-K filings and balance sheets. They contain quantitative valuation metrics (like P/E and P/S ratios), liquidity metrics, Return on Equity (ROE), and exact profit margins.

   By keeping these distinct, the RAG system must successfully combine qualitative narrative with quantitative metrics to generate a holistic Wall-Street-grade investment brief!

---

## 5. How to Run

### Step 1: Data Ingestion
Ingest the financial documents into the ChromaDB vector database. You only need to run this once or whenever you add new text files to the data directory.
```bash
python ingest.py
```
*Expected Output: The script will connect to your local Ollama embedding model and report how many documents were successfully added to the ChromaDB collection.*

### Step 2: Run the Multi-Agent Loop
Run the main script and specify the company you want to research.
```bash
python main.py --company "Apple"
```
*Expected Output: The terminal will print out the conversational transcript between the `ResearchAgent` and the `CriticAgent`. Once the CriticAgent approves the brief, the final output will be saved to the `output/` directory as `brief_{company}.txt`.*

---

## 6. The Full Pipeline Explained

Here is the exact step-by-step workflow of what happens when you run `python main.py --company "Apple"`. This flow embodies the core philosophy of **AutoGen orchestration**:

1. **User Initiation (CLI):** The user triggers the script and passes the target company name (`Apple`).
2. **RAG Context Retrieval (`retriever.py`):** 
   - The script connects to the local ChromaDB.
   - It embeds specific queries (e.g., "Apple revenue growth") using the local `nomic-embed-text` model.
   - It performs semantic similarity search against the previously ingested financial documents.
   - The top matching paragraphs are formatted into a single string.
3. **AutoGen Loop Initiation (`main.py` & `agents.py`):**
   - The system constructs a starting prompt combining your instructions and the retrieved ChromaDB string.
   - The `CriticAgent` formally "initiates the chat" by sending this starting prompt to the `ResearchAgent`.
4. **Agent Drafting (Turn 1):** 
   - The `ResearchAgent` reads the prompt and drafts the first version of the investment brief, attempting to include executive summaries, metrics, and risks based *only* on the provided context.
5. **Agent Critique (Turn 2):** 
   - The draft is sent back to the `CriticAgent`. 
   - The `CriticAgent` evaluates the text against its strict 5-point system prompt checklist. 
   - If the brief is missing a specific number or uses vague language, the `CriticAgent` writes a critique demanding a fix and passes the turn back to the `ResearchAgent`.
6. **Iterative Refinement (Turns 3+):** 
   - The agents continue passing the document back and forth. The ResearchAgent updates the text, and the CriticAgent re-evaluates it.
7. **Termination & Output (`main.py`):** 
   - Once the `CriticAgent` is perfectly satisfied, it replies with the exact string `"APPROVED"`. 
   - The custom `is_approved` callback function in AutoGen detects this keyword and shuts down the conversation loop.
   - The final accepted text is parsed out of the AutoGen chat history array and saved to your hard drive.

## 7. In-Depth Python Scripts and Function Definitions

### `ingest.py`
This one-time setup script processes raw text files into vector embeddings for semantic search.

```python
import os
import glob
import chromadb
from chromadb.utils import embedding_functions

def main():
    print("Starting document ingestion pipeline...")
    
    # 1. Load the Ollama embedding function
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
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(
            name="financial_docs", 
            embedding_function=ollama_ef
        )
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return

    # 3. Read documents from the /data/financial/ directory
    data_dir = "./data/financial/"
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
                
            filename = os.path.basename(filepath)
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('_')
            
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
            ids.append(f"doc_{filename}")
            
            print(f"Loaded: {filename} (Company: {company})")
            
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")

    # 4. Add the documents to the ChromaDB collection
    if documents:
        try:
            print("Adding documents to ChromaDB...")
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
```
*Explanation:* We connect to our local Ollama instance to use the `nomic-embed-text` model. This function converts raw text into mathematical vectors. We then connect to ChromaDB (or create it if it doesn't exist). `upsert` ensures that if we run the script twice, it updates the existing documents rather than duplicating them. The `metadatas` dictionary allows us to filter searches later (e.g., only search files where `company == "apple"`).

### `retriever.py`
This script represents the "Retrieval" in RAG.

```python
import chromadb
from chromadb.utils import embedding_functions

def retrieve_financial_context(company: str, year: str = None) -> str:
    """
    Queries ChromaDB with the company name (and optional year) and financial sub-queries to retrieve context.
    Returns a formatted string of the top-k relevant document chunks.
    """
    try:
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text:latest"
        )
        
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(
            name="financial_docs",
            embedding_function=ollama_ef
        )
    except Exception as e:
        print(f"Error accessing ChromaDB: {e}")
        return "Error: Could not retrieve context due to database access issues."
        
    queries = [
        f"{company} revenue growth earnings",
        f"{company} financial ratios debt margins",
        f"{company} risk factors headwinds"
    ]
    
    retrieved_chunks = []
    seen_docs = set()
    
    try:
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
                n_results=3,
                where=where_clause 
            )
            
            if results and 'documents' in results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    doc_id = results['ids'][0][i]
                    metadata = results['metadatas'][0][i]
                    
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
        
    full_context = "\n".join(retrieved_chunks)
    return full_context

if __name__ == "__main__":
    print("Testing retriever for Apple...")
    print(retrieve_financial_context("Apple"))
```
*Explanation:* Instead of sending one massive query, we send multiple hyper-specific sub-queries (revenue, risk, ratios). The `where` clause is a deterministic metadata filter, ensuring we never accidentally pull Google's data when researching Apple. Because multiple sub-queries might return the exact same document chunk, we use a Python `set()` to track `doc_id`s. This prevents duplicate paragraphs from eating up the `ResearchAgent`'s context window limits.

### `agents.py`
This script constructs the intelligence and personas of the multi-agent system.

```python
import autogen

# Configuration for Ollama
config_list = [{
    "model": "llama3.1:latest",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
}]

llm_config = {
    "config_list": config_list,
    "temperature": 0.3,
}

def is_approved(msg):
    """Termination condition: Check if the exact word 'APPROVED' is in the message."""
    content = msg.get("content", "")
    if content is None:
        return False
    return "APPROVED" in content

research_agent = autogen.AssistantAgent(
    name="ResearchAgent",
    system_message="""You are a meticulous financial Research Analyst.
Your task is to write a detailed, 300-word minimum investment brief for the specified company.

CRITICAL INSTRUCTIONS:
1. You MUST use the provided financial context to draft the brief. Do not hallucinate data.
2. Structure your output with the following exact sections:
   - Executive Summary
   - Key Financial Metrics
   - Risk Factors
   - Growth Catalysts
   - Preliminary Recommendation
3. You MUST cite specific numbers, figures, and ratios from the context (e.g., "FY2023 gross margin of 44.1%"). Do not use vague qualitative language when specific data is available.
4. When the CriticAgent provides feedback, you MUST incorporate the specific requested changes into your revised brief. Do not simply restate your original draft.
""",
    llm_config=llm_config,
    max_consecutive_auto_reply=3,
    is_termination_msg=is_approved,
)

critic_agent = autogen.AssistantAgent(
    name="CriticAgent",
    system_message="""You are a strict, senior financial Critic Manager.
Your job is to review the draft investment brief against a rigorous checklist.

CHECKLIST:
1. Are all financial claims supported by specific figures from the retrieved documents?
2. Does the brief include a dedicated risk factor section with at least 3 identified risks (if present in the context)?
3. Is the recommendation logically supported by the evidence presented?
4. Does the brief meet the 300-word minimum?
5. Is there any vague qualitative language where retrieved data provides specifics?

CRITICAL INSTRUCTIONS:
- If the brief FAILS any checklist item, you must return specific, actionable fix requests. Identify the exact claim that is weak and state what specific data from the context should replace it.
- Do NOT give general feedback like "improve the analysis".
- If and ONLY if the brief passes ALL checklist items, you must output exactly the word "APPROVED" on a single line with no other text.
""",
    llm_config=llm_config,
    max_consecutive_auto_reply=3,
    is_termination_msg=is_approved,
)
```
*Explanation:* We configure AutoGen to spoof an OpenAI API connection, rerouting it to our local Ollama server. The `temperature` is strictly set to `0.3` to minimize creative hallucinations, forcing the model to be highly deterministic and factual. We instantiate two distinct agents and inject our custom `is_approved` function directly into their `is_termination_msg` attribute so they know exactly when to stop. The Critic's prompt is heavily engineered to act as an adversarial gatekeeper.

### `main.py`
This script orchestrates the pipeline, tying RAG and AutoGen together.

```python
import argparse
import os
from datetime import datetime
from retriever import retrieve_financial_context
from agents import research_agent, critic_agent
import autogen

def main():
    parser = argparse.ArgumentParser(description="Run the AutoGen Investment Research Assistant")
    parser.add_argument("--company", type=str, required=True, help="The company name to research (e.g., 'Apple')")
    parser.add_argument("--year", type=str, required=True, help="Specific year to filter data (e.g., '2024')")
    args = parser.parse_args()
    company = args.company
    year = args.year
    
    print(f"Starting research on: {company}" + (f" for year {year}" if year else ""))
    
    try:
        print("Retrieving financial context from ChromaDB...")
        context = retrieve_financial_context(company, year)
    except Exception as e:
        print(f"Failed to retrieve context: {e}")
        return

    if context.startswith("Error") or context.startswith("No financial documents"):
        print(context)
        print("Please ensure you have run ingest.py and the data exists.")
        return

    initial_message = f"""Please draft an investment brief for {company}.

Here is the retrieved financial context you MUST use:
{context}

Remember to follow your system instructions carefully, structure the brief as required, and aim for a 300-word minimum length. Include specific figures.
"""

    print("Initiating agent conversation loop...\n")
    try:
        chat_result = critic_agent.initiate_chat(
            research_agent,
            message=initial_message,
            summary_method="last_msg"
        )
    except Exception as e:
        print(f"Error during AutoGen conversation: {e}")
        return

    print("\nConversation finished. Processing output...")
    
    chat_history = chat_result.chat_history
    final_brief = "No brief generated."
    rounds = len(chat_history) // 2 
    
    for msg in reversed(chat_history):
        if msg.get("name") == "ResearchAgent":
            content = msg.get("content", "")
            # Verify this is the actual brief and not a meta-response
            if "Executive Summary" in content or len(content) > 150:
                final_brief = content
                break
            
    is_success = False
    if chat_history and "APPROVED" in chat_history[-1].get("content", ""):
        is_success = True

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"brief_{company.lower()}_{year}.txt"
    filepath = os.path.join(output_dir, filename)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_str = "APPROVED" if is_success else "NOT APPROVED (Max Rounds Reached)"
    
    header = f"""================================================================================
INVESTMENT BRIEF — {company.upper()}
Generated: {timestamp}
Status: {status_str}
Rounds to finish: {rounds}
================================================================================

"""
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(final_brief)
        print(f"\nBrief successfully saved to: {filepath}")
    except Exception as e:
        print(f"Error saving brief to file: {e}")

if __name__ == "__main__":
    main()
```
*Explanation:* The `is_approved` function is passed to the `is_termination_msg` parameter in AutoGen. AutoGen intercepts every message in the chat loop and runs this function. If it evaluates to `True`, the conversation halts immediately. We bridge RAG and AutoGen by injecting the retrieved `context` directly into the starting prompt. The `CriticAgent` then sends this payload to the `ResearchAgent`, officially kicking off the multi-turn conversational loop. Once the chat finishes, AutoGen returns a dictionary containing the entire history. We iterate through this list backwards to find the *most recent* (and therefore most refined) message sent by the `ResearchAgent` to save it to disk.

---

## 8. Bonus Challenges

Once you have the core AutoGen pipeline working, you can attempt the following bonus challenges to level up your system. Here is exactly how to implement them and how they will alter your architecture.

### Bonus A: The FormatAgent (JSON Parsing)
**Goal:** In real-world applications, raw text briefs are hard to insert into databases or display on web dashboards. You need to convert the final, approved text into structured data.
**Pipeline Impact:** Instead of the pipeline ending when the `CriticAgent` approves the brief, we will inject a *third* agent that acts as a downstream data processor.

**How to Implement (Step-by-Step):**
1. **Where:** `agents.py`
   **What:** Define a third agent: `format_agent = autogen.AssistantAgent(name="FormatAgent", ...)`
   **Prompt Modification:** Give it a strict system prompt: *"You are a data extraction bot. You will receive a text brief. You must strictly output a raw JSON object and nothing else. The JSON must contain exactly these keys: {company, rating, key_risks, recommendation}."*
2. **Where:** `main.py`
   **What:** Right now, when the `CriticAgent` loop successfully finishes, it extracts `final_brief` and writes it to a `.txt` file. You need to interrupt this save process.
3. **Pipeline Change:** Instead of saving immediately, initialize a *second* AutoGen chat right below the first one. Create a dummy `autogen.UserProxyAgent` that sends the `final_brief` text string directly to the `format_agent`.
4. **Final Step:** Extract the `format_agent`'s response, use Python's built-in `json.loads()` to verify it is valid JSON, and then save it to disk as `output/brief_{company}.json`.

### Bonus B: Dynamic Web Search Tool (DuckDuckGo)
**Goal:** Our RAG pipeline currently relies purely on static `.txt` files in `data/financial/`. What if a major news event happened *this morning*? You need to equip the `ResearchAgent` with live internet access.
**Pipeline Impact:** Instead of the `ResearchAgent` immediately drafting the brief based *only* on the injected RAG context from ChromaDB, the agent will pause, execute a tool call (triggering a local Python function), receive live internet data, and *then* draft the brief by combining the historical RAG data with live breaking news.

**How to Implement (Step-by-Step):**
1. **Where:** Terminal
   **What:** Install the search library: `pip install duckduckgo-search`
2. **Where:** A new file called `tools.py` (or directly in `agents.py`)
   **What:** Write a pure Python function that queries the internet and returns a string of news:
   ```python
   from duckduckgo_search import DDGS
   def search_news(query: str) -> str:
       results = DDGS().text(f"{query} financial news", max_results=3)
       return str(results)
   ```
3. **Where:** `agents.py`
   **What:** You must explicitly bind this tool to the agent so the LLM knows it exists and how to execute it. Use AutoGen's `register_function`:
   ```python
   from autogen import register_function
   # (Note: In AutoGen, you typically need a UserProxyAgent to act as the tool 'executor')
   register_function(
       search_news,
       caller=research_agent,
       executor=tool_executor_agent, 
       name="search_news",
       description="Search the web for real-time news about a company."
   )
   ```
4. **Where:** `agents.py`
   **What:** Update the `ResearchAgent`'s `system_message` to explicitly instruct it: *"Before writing the brief, you MUST use the `search_news` tool to check for breaking news. Incorporate any breaking news into the 'Risk Factors' or 'Growth Catalysts' sections."*

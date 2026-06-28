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

    # Check if context is an error message
    if context.startswith("Error") or context.startswith("No financial documents"):
        print(context)
        print("Please ensure you have run ingest.py and the data exists.")
        return

    # 3. Formulate the initial message embedding the retrieved context
    initial_message = f"""Please draft an investment brief for {company}.

Here is the retrieved financial context you MUST use:
{context}

Remember to follow your system instructions carefully, structure the brief as required, and aim for a 300-word minimum length. Include specific figures.
"""

    # 4. Initiate the AutoGen conversation loop
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

    # 5 & 6. Handle output and extract the final brief
    print("\nConversation finished. Processing output...")
    
    chat_history = chat_result.chat_history
    final_brief = "No brief generated."
    rounds = len(chat_history) // 2  # Approximate rounds
    
    for msg in reversed(chat_history):
        if msg.get("name") == "ResearchAgent":
            content = msg.get("content", "")
            # Verify this is the actual brief and not a meta-response
            if "Executive Summary" in content or len(content) > 150:
                final_brief = content
                break
            
    # Check if the loop terminated successfully or due to max replies
    is_success = False
    if chat_history and "APPROVED" in chat_history[-1].get("content", ""):
        is_success = True

    # 7. Write the brief to output/brief_{company}.txt with a timestamp header
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

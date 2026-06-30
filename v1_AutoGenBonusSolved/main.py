import argparse
import os
from datetime import datetime
from retriever import retrieve_financial_context
from agents import research_agent, critic_agent, format_agent, format_proxy
import autogen
import json

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

    # ---------------------------------------------------------------------
    # BONUS A: JSON Parsing Phase
    # ---------------------------------------------------------------------
    if is_success:
        print("\nStarting JSON formatting phase...")
        try:
            format_chat = format_proxy.initiate_chat(
                format_agent,
                message=f"Convert the following investment brief into JSON according to your system prompt schema:\n\n{final_brief}",
                summary_method="last_msg"
            )
            
            # The format_agent's response should be the last message in the chat
            json_response = format_chat.chat_history[-1].get("content", "")
            
            # Clean up the JSON string (small models often add markdown or forget the final brace)
            json_response = json_response.strip()
            if json_response.startswith("```json"):
                json_response = json_response[7:]
            elif json_response.startswith("```"):
                json_response = json_response[3:]
            if json_response.endswith("```"):
                json_response = json_response[:-3]
            json_response = json_response.strip()
            
            # If the model truncated the output, append a closing brace
            if json_response.startswith("{") and not json_response.endswith("}"):
                if not json_response.endswith('"'):
                    json_response += '"'
                json_response += "}"
            
            # Try to parse it to ensure it's valid JSON, then save it
            try:
                parsed_json = json.loads(json_response)
                json_filename = f"brief_{company.lower()}_{year}.json"
                json_filepath = os.path.join(output_dir, json_filename)
                
                with open(json_filepath, "w", encoding="utf-8") as jf:
                    json.dump(parsed_json, jf, indent=4)
                print(f"Successfully converted and saved structured JSON to: {json_filepath}")
            except json.JSONDecodeError:
                print("Error: The FormatAgent failed to output valid JSON.")
                print(f"Raw Output:\n{json_response}")
                
        except Exception as e:
            print(f"Error during JSON formatting phase: {e}")

if __name__ == "__main__":
    main()

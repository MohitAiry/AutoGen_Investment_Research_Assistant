import autogen
from tools import search_news

# Configuration for Ollama
config_list = [{
    "model": "llama3.2:3b",  # Using a faster model to avoid long wait times
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
}]

llm_config = {
    "config_list": config_list,
    "temperature": 0.1, # Low temp for data accuracy
}

def is_approved(msg):
    """Termination condition: Check if the exact word 'APPROVED' is in the message."""
    content = msg.get("content", "")
    if content is None:
        return False
    return "APPROVED" in content

# The ResearchAgent drafts the brief based on retrieved context
# We use max_consecutive_auto_reply=3 to prevent infinite loops if they can't agree
research_agent = autogen.AssistantAgent(
    name="ResearchAgent",
    system_message="""You are a meticulous financial Research Analyst.
Your task is to write a detailed, 300-word minimum investment brief for the specified company.

CRITICAL INSTRUCTIONS:
1. You MUST use the provided financial context to draft the brief. Do not hallucinate data.
2. Structure your output with the following exact sections:
   - Executive Summary
   - Key Financial Metrics [MUST include at least 5 bullet points with EXACT NUMBERS from the ratios document]
   - Risk Factors [Integrate live news here]
   - Growth Catalysts [Integrate live news here]
   - Preliminary Recommendation
3. You MUST cite specific numbers, figures, and ratios from the context. Do not use vague qualitative language when specific data is available.
4. When the CriticAgent provides feedback, you MUST incorporate the specific requested changes into your revised brief. Do not simply restate your original draft.
5. You have access to a web search tool called `search_news`. Before writing the brief, you MUST use this tool to search the live web for breaking news about the company. Incorporate any breaking news into the 'Risk Factors' or 'Growth Catalysts' sections. If the tool returns no news, DO NOT call it again; proceed immediately to writing the brief.
""",
    llm_config=llm_config,
    max_consecutive_auto_reply=8,
    is_termination_msg=is_approved,
)

# The CriticAgent reviews the brief and provides actionable feedback
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
    max_consecutive_auto_reply=8,
    is_termination_msg=is_approved,
)

# -------------------------------------------------------------------------
# BONUS A: The FormatAgent (JSON Parsing)
# -------------------------------------------------------------------------
# This agent's only job is to take the final approved brief and convert it into JSON.
format_agent = autogen.AssistantAgent(
    name="FormatAgent",
    system_message="""You are a strict data extraction bot. 
Your ONLY job is to take a raw text investment brief and output a valid JSON object.
Do not output any markdown formatting, backticks, or conversational text. Output ONLY the raw JSON string.
The JSON must strictly follow this schema:
{
  "ticker": "COMPANY_NAME",
  "recommendation": "Buy/Hold/Sell",
  "key_risks": ["risk 1", "risk 2"],
  "summary": "executive summary text here",
  "metrics": "key financial metrics summary"
}
""",
    llm_config=llm_config,
)

# We need a UserProxy to send the final text to the FormatAgent
format_proxy = autogen.UserProxyAgent(
    name="FormatProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,  # We only need one turn: Proxy sends text -> FormatAgent returns JSON
    code_execution_config=False,   # Disable Docker execution since we only need text parsing
)

# -------------------------------------------------------------------------
# BONUS B: Dynamic Web Search Tool Registration
# -------------------------------------------------------------------------
# We register the tool so the ResearchAgent knows it exists, and the CriticAgent knows how to execute it.
autogen.agentchat.register_function(
    search_news,
    caller=research_agent,
    executor=critic_agent,
    name="search_news",
    description="Search the web for real-time news about a company. Input should be the company name."
)

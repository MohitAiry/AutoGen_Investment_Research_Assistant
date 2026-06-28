import autogen

# Configuration for Ollama
config_list = [{
    "model": "llama3.1:latest",  # Note: Ensure this matches the model you have in ollama (e.g. 'llama3')
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",  # Placeholder, not actually used by Ollama local API
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
    max_consecutive_auto_reply=3,
    is_termination_msg=is_approved,
)

import streamlit as st
import subprocess
import os
import json

st.set_page_config(page_title="Agentic Research Assistant", layout="wide", page_icon="📈")

st.title("🤖 Autonomous AI Research Agent")
st.markdown("Enter a ticker symbol below. The Research and Critic agents will autonomously analyze the company and generate a structured investment brief using live financial data and real-time news.")

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configuration")
    ticker = st.text_input("Ticker Symbol", value="AAPL", help="e.g., AAPL, MSFT, TSLA, NVDA")
    year = st.text_input("Financial Year", value="2024", help="Data is fetched live from yfinance and tagged with this year.")
    
    generate_btn = st.button("🚀 Generate Investment Brief", type="primary", use_container_width=True)

    if generate_btn:
        st.session_state['running'] = True

with col2:
    if st.session_state.get('running', False):
        st.subheader("Agent Conversation Log")
        
        with st.spinner(f"Agents are actively researching {ticker}... Please wait (this can take 1-2 minutes on local models)."):
            # Run main.py as a subprocess
            result = subprocess.run(
                ["python", "main.py", "--company", ticker, "--year", year], 
                capture_output=True, 
                text=True
            )
            
            st.text_area("Terminal Output", result.stdout, height=300)
            
            if result.stderr:
                st.error("Errors encountered during execution:")
                st.code(result.stderr)
                
            st.session_state['run_complete'] = True
            st.session_state['running'] = False
            st.session_state['ticker'] = ticker
            st.session_state['year'] = year

st.markdown("---")

if st.session_state.get('run_complete', False):
    st.header("Final Output")
    
    t_ticker = st.session_state['ticker']
    t_year = st.session_state['year']
    
    brief_path = f"output/brief_{t_ticker.lower()}_{t_year}.txt"
    json_path = f"output/brief_{t_ticker.lower()}_{t_year}.json"
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 📝 Written Brief")
        if os.path.exists(brief_path):
            with open(brief_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.warning("No written brief was generated.")
            
    with col4:
        st.markdown("### 📊 Structured Data (JSON)")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    st.json(data)
                except Exception as e:
                    st.error("Generated JSON was malformed.")
                    st.code(f.read())
        else:
            st.warning("No JSON data was generated. The agents might not have reached an APPROVED consensus.")

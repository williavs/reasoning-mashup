import streamlit as st
import asyncio
from datetime import datetime

from api.simple_flow import ResearchAgent
from api.research_workflow import ResearchWorkflow
from api.sales_qualification_workflow import SalesQualificationWorkflow
from api.proxy_config import ProxyAPIConfig

# Custom progress logger for Streamlit
class StreamlitLogger:
    def __init__(self, container):
        self.container = container
        self.logs = []
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        # Update the container with all logs
        self.container.code('\n'.join(self.logs), language='bash')

st.set_page_config(
    page_title="Research Analysis Assistant",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS to improve markdown rendering
st.markdown("""
<style>
.stMarkdown {
    max-width: 100% !important;
}
.research-results {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f8f9fa;
    margin: 1rem 0;
}
.debug-output {
    background-color: #1e1e1e;
    color: #d4d4d4;
    padding: 1rem;
    border-radius: 0.5rem;
    font-family: monospace;
    margin: 1rem 0;
    max-height: 300px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

st.title("Research Analysis Assistant 🔬")

# Initialize session states
if 'workflow_type' not in st.session_state:
    st.session_state.workflow_type = 'simple'
if 'api_type' not in st.session_state:
    st.session_state.api_type = ProxyAPIConfig.OPENROUTER
if 'simple_agent' not in st.session_state:
    st.session_state.simple_agent = ResearchAgent()
if 'multi_agent_workflow' not in st.session_state:
    st.session_state.multi_agent_workflow = None
if 'sales_workflow' not in st.session_state:
    st.session_state.sales_workflow = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for workflow selection and settings
with st.sidebar:
    st.title("Settings")
    
    # API selector
    api_type = st.radio(
        "Select API Backend",
        [ProxyAPIConfig.OPENROUTER, ProxyAPIConfig.OLLAMA],
        index=0 if st.session_state.api_type == ProxyAPIConfig.OPENROUTER else 1,
        help="Choose between OpenRouter (Claude) or Ollama for LLM processing"
    )
    
    # Update API type in session state
    if api_type != st.session_state.api_type:
        st.session_state.api_type = api_type
        # Reset workflows when API changes
        st.session_state.multi_agent_workflow = None
        st.session_state.sales_workflow = None
        st.session_state.simple_agent = ResearchAgent(api_type=api_type)
    
    # Workflow selector
    workflow_type = st.radio(
        "Select Workflow Type",
        ['Simple Agent', 'Multi-Agent Workflow', 'Sales Qualification'],
        index=0 if st.session_state.workflow_type == 'simple' else 
              1 if st.session_state.workflow_type == 'multi' else 2,
        help="Choose between a single research agent, specialized multi-agent workflow, or sales qualification workflow"
    )
    
    # Update workflow type in session state
    st.session_state.workflow_type = ('simple' if workflow_type == 'Simple Agent' 
                                    else 'multi' if workflow_type == 'Multi-Agent Workflow' 
                                    else 'sales')
    
    # Debug mode toggle
    show_debug = st.toggle("Show Debug Output", value=True)
    
    st.divider()
    
    # About section based on selected workflow and API
    st.title("About")
    if st.session_state.workflow_type == 'simple':
        st.markdown(f"""
        **Simple Agent Workflow** using {st.session_state.api_type.title()}
        
        This workflow uses a single research agent to:
        1. Search the internet for relevant information
        2. Analyze the current state of research
        3. Identify challenges and opportunities
        4. Suggest future directions
        5. Highlight practical applications
        """)
    elif st.session_state.workflow_type == 'multi':
        st.markdown(f"""
        **Multi-Agent Workflow** using {st.session_state.api_type.title()}
        
        This workflow uses specialized agents:
        1. 📚 Literature Reviewer
        2. 🔍 Gap Analyzer
        3. 🧪 Methodology Expert
        4. 📊 Impact Assessor
        5. 📝 Research Synthesizer
        
        Each agent builds on the previous agent's analysis.
        """)
    else:
        st.markdown(f"""
        **Sales Qualification Workflow** using {st.session_state.api_type.title()}
        
        This workflow analyzes potential clients using specialized agents:
        1. 🏢 Company Research
        2. 📋 Compliance Analysis
        3. 🔄 Competitor Analysis
        4. ✅ Fit Assessment
        5. 📝 Final Synthesis
        
        Provides comprehensive qualification analysis for sales teams.
        """)
    
    st.divider()
    
    # Example topics/companies based on workflow
    st.subheader("Example Topics" if st.session_state.workflow_type != 'sales' else "Example Companies")
    
    if st.session_state.workflow_type == 'sales':
        example_companies = [
            "Acme Startup Inc",
            "TechGrowth Solutions",
            "Modern Retail Group",
            "Digital Services Co"
        ]
        for company in example_companies:
            if st.button(f"Try: {company}"):
                st.chat_input(company)
                st.rerun()
    else:
        example_topics = [
            "quantum computing applications",
            "CRISPR gene editing advances",
            "renewable energy storage solutions",
            "artificial general intelligence progress"
        ]
        for topic in example_topics:
            if st.button(f"Try: {topic}"):
                st.chat_input(topic)
                st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your research topic..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show assistant response
    with st.chat_message("assistant"):
        with st.status("🔍 Analysis in Progress...", expanded=True) as status:
            # Create debug output container
            debug_container = st.empty()
            logger = StreamlitLogger(debug_container)
            
            if st.session_state.workflow_type == 'simple':
                # Simple Agent Workflow
                logger.log("🌐 Initializing simple agent workflow...")
                progress_placeholder = st.empty()
                progress_placeholder.markdown("🌐 Gathering internet research...")
                
                result = asyncio.run(st.session_state.simple_agent.process_topic(prompt))
                
                if result:
                    logger.log(f"✅ Analysis complete in {result.get('elapsed_time', 'N/A')}s")
                    
                    response_md = f"""
                    <div class="research-results">
                    ### 📊 Research Analysis Results
                    
                    ⏱️ Time Taken: {result.get('elapsed_time', 'N/A')}s
                    
                    ### 💭 Reasoning Process
                    {result.get('reasoning', 'No reasoning provided')}
                    
                    ### 🔍 Key Findings
                    {result.get('response', 'No response provided')}
                    </div>
                    """
                    
                    progress_placeholder.empty()
                    st.markdown(response_md, unsafe_allow_html=True)
                    status.update(label="✅ Analysis Complete", state="complete")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_md
                    })
                else:
                    error_message = "❌ Analysis failed. Please try again."
                    progress_placeholder.error(error_message)
                    status.update(label="❌ Analysis Failed", state="error")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
            elif st.session_state.workflow_type == 'multi':
                # Multi-Agent Research Workflow
                logger.log("🤖 Initializing multi-agent workflow...")
                progress_placeholder = st.empty()
                
                # Initialize workflow with logger if needed
                if st.session_state.multi_agent_workflow is None:
                    st.session_state.multi_agent_workflow = ResearchWorkflow(api_type=st.session_state.api_type, logger=logger.log)
                
                results = asyncio.run(st.session_state.multi_agent_workflow.process_topic(prompt))
                
                if results:
                    # Format multi-agent results
                    response_md = "<div class='research-results'>"
                    total_time = 0
                    
                    for step, result in results.items():
                        if "error" in result:
                            response_md += f"\n### ❌ {step}\n{result['error']}\n"
                        else:
                            time = result.get('time', 0)
                            total_time += time
                            response_md += f"""
                            ### 📋 {step}
                            
                            ⏱️ Time: {time}s
                            
                            #### 💭 Reasoning
                            {result['reasoning']}
                            
                            #### 🔍 Analysis
                            {result['response']}
                            
                            ---
                            """
                    
                    response_md += f"\n### ⌛ Total Analysis Time: {total_time}s"
                    response_md += "</div>"
                    
                    progress_placeholder.empty()
                    st.markdown(response_md, unsafe_allow_html=True)
                    status.update(label="✅ Analysis Complete", state="complete")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_md
                    })
                else:
                    error_message = "❌ Analysis failed. Please try again."
                    progress_placeholder.error(error_message)
                    status.update(label="❌ Analysis Failed", state="error")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
            else:
                # Sales Qualification Workflow
                logger.log("🤖 Initializing sales qualification workflow...")
                progress_placeholder = st.empty()
                
                # Initialize workflow with logger if needed
                if st.session_state.sales_workflow is None:
                    st.session_state.sales_workflow = SalesQualificationWorkflow(api_type=st.session_state.api_type, logger=logger.log)
                
                results = asyncio.run(st.session_state.sales_workflow.process_company(prompt))
                
                if results:
                    # Format sales qualification results
                    response_md = "<div class='research-results'>"
                    total_time = 0
                    
                    for step, result in results.items():
                        if "error" in result:
                            response_md += f"\n### ❌ {step}\n{result['error']}\n"
                        else:
                            time = result.get('time', 0)
                            total_time += time
                            response_md += f"""
                            ### 📋 {step}
                            
                            ⏱️ Time: {time}s
                            
                            #### 💭 Analysis Process
                            {result['reasoning']}
                            
                            #### 🔍 Key Findings
                            {result['response']}
                            
                            ---
                            """
                    
                    response_md += f"\n### ⌛ Total Analysis Time: {total_time}s"
                    response_md += "</div>"
                    
                    progress_placeholder.empty()
                    st.markdown(response_md, unsafe_allow_html=True)
                    status.update(label="✅ Analysis Complete", state="complete")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_md
                    })
                else:
                    error_message = "❌ Analysis failed. Please try again."
                    progress_placeholder.error(error_message)
                    status.update(label="❌ Analysis Failed", state="error")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    }) 
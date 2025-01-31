from praisonaiagents import Agent, Task, PraisonAIAgents
import asyncio
import requests
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from typing import List, Dict, Optional, Callable
from .proxy_config import ProxyAPITool, ProxyAPIConfig

# Load environment variables
load_dotenv()

def internet_search_tool(query: str) -> List[Dict]:
    """DuckDuckGo search tool"""
    results = []
    try:
        ddgs = DDGS()
        for result in ddgs.text(keywords=query, max_results=5):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("body", "")
            })
    except Exception as e:
        print(f"⚠️ Search error: {str(e)}")
    return results

class CompanyResearchAgent(Agent):
    """Agent specialized in comprehensive company research and analysis"""
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER, logger: Optional[Callable] = None):
        self.logger = logger or print
        super().__init__(
            name="Company Intelligence Gatherer",
            role="Corporate Intelligence Analyst",
            goal="Gather and analyze comprehensive company information including history, operations, financials, team, and recent developments",
            backstory="Expert in deep corporate research with access to extensive business intelligence resources",
            tools=[internet_search_tool],
            llm=ProxyAPITool(api_type=api_type, logger=logger)
        )

class ComplianceAnalysisAgent(Agent):
    """Agent specialized in regulatory and operational analysis"""
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER, logger: Optional[Callable] = None):
        self.logger = logger or print
        super().__init__(
            name="Operations Analyzer",
            role="Business Operations Specialist",
            goal="Research and document company's operational structure, locations, regulatory environment, and business model",
            backstory="Expert in business operations analysis and regulatory frameworks",
            tools=[internet_search_tool],
            llm=ProxyAPITool(api_type=api_type, logger=logger)
        )

class CompetitorAnalysisAgent(Agent):
    """Agent specialized in market and ecosystem analysis"""
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER, logger: Optional[Callable] = None):
        self.logger = logger or print
        super().__init__(
            name="Market Intelligence Analyst",
            role="Market Research Specialist",
            goal="Map out the company's complete business ecosystem, partnerships, competitors, and market position",
            backstory="Expert in competitive intelligence and market ecosystem analysis",
            tools=[internet_search_tool],
            llm=ProxyAPITool(api_type=api_type, logger=logger)
        )

class FitAssessmentAgent(Agent):
    """Agent specialized in comprehensive business analysis"""
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER, logger: Optional[Callable] = None):
        self.logger = logger or print
        super().__init__(
            name="Business Analyst",
            role="Strategic Business Analyst",
            goal="Synthesize all gathered information into a comprehensive company profile",
            backstory="Expert in business analysis and strategic intelligence synthesis",
            tools=[internet_search_tool],
            llm=ProxyAPITool(api_type=api_type, logger=logger)
        )

class SalesQualificationWorkflow:
    """Orchestrates the multi-agent company research workflow"""
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER, logger: Optional[Callable] = None):
        self.logger = logger or print
        self.api_type = api_type
        
        # Initialize agents with logger and API type
        self.company_agent = CompanyResearchAgent(api_type=api_type, logger=logger)
        self.compliance_agent = ComplianceAnalysisAgent(api_type=api_type, logger=logger)
        self.competitor_agent = CompetitorAnalysisAgent(api_type=api_type, logger=logger)
        self.fit_agent = FitAssessmentAgent(api_type=api_type, logger=logger)
        
        # Create tasks for each agent
        self.tasks = [
            Task(
                description="Conduct comprehensive company research",
                expected_output="Detailed company intelligence report",
                agent=self.company_agent
            ),
            Task(
                description="Analyze operations and regulatory landscape",
                expected_output="Operations and regulatory analysis report",
                agent=self.compliance_agent
            ),
            Task(
                description="Research market ecosystem and positioning",
                expected_output="Market intelligence report",
                agent=self.competitor_agent
            ),
            Task(
                description="Synthesize comprehensive company profile",
                expected_output="Complete business analysis report",
                agent=self.fit_agent
            )
        ]
    
    async def process_company(self, company_name: str) -> Dict:
        """Process a company through the comprehensive research workflow"""
        self.logger(f"\n🔍 Starting comprehensive company analysis for: {company_name}")
        self.logger("=" * 50)
        
        results = {}
        context = ""
        
        for i, task in enumerate(self.tasks, 1):
            self.logger(f"\n📋 Step {i}: {task.description}")
            self.logger(f"🤖 Agent: {task.agent.name}")
            
            # Prepare prompt with context from previous steps
            prompt = f"""Target Company: {company_name}

Previous Analysis:
{context}

Your Task: {task.description}
As a {task.agent.role}, conduct exhaustive research and provide:

1. COMPREHENSIVE ANALYSIS:
   - Company Overview: history, mission, vision
   - Leadership Team: key executives, backgrounds
   - Financial Information: revenue, funding, growth metrics
   - Product/Service Portfolio: detailed offerings
   - Technology Stack: systems and tools used
   - Recent News: developments, announcements, press coverage
   - Market Position: industry standing, market share
   - Customer Base: target markets, key clients
   - Partnerships: strategic relationships, integrations
   - Growth Trajectory: expansion plans, historical growth
   - Company Culture: values, employee reviews, workplace
   - Legal/Regulatory Status: compliance, licenses, permits
   - Geographic Presence: office locations, market reach
   - Competitive Advantages: unique selling propositions
   - Challenges: identified issues, market obstacles

2. KEY FINDINGS:
   - Most significant discoveries
   - Unique insights
   - Critical data points
   - Notable trends

3. INFORMATION SOURCES:
   - List all sources used
   - Credibility assessment
   - Data freshness/timeliness
   - Information gaps identified

Focus on gathering and presenting ALL available accurate information. Be thorough and exhaustive in your research. Include both positive and negative findings. Cite sources where possible.

If you find conflicting information, present all versions with source context. If you're unsure about something, explicitly state it rather than making assumptions.

Be analytical and fact-focused in your response."""

            # Execute task
            self.logger(f"🎯 Executing task: {task.description}")
            result = await task.agent.llm.execute(prompt)
            
            if result:
                self.logger(f"✅ Step {i} complete")
                # Add to results
                results[task.description] = {
                    "reasoning": result.get("reasoning", ""),
                    "response": result.get("response", ""),
                    "time": result.get("elapsed_time", 0)
                }
                # Update context for next step
                context += f"\n\nPrevious Step ({task.agent.name}):\n{result.get('response', '')}"
                self.logger(f"⏱️  Time taken: {result.get('elapsed_time', 0)}s")
            else:
                self.logger(f"❌ Step {i} failed")
                results[task.description] = {"error": "Task failed"}
        
        return results

async def main():
    # Get company name
    company_name = input("\n🎯 Enter company name to qualify: ")
    
    # Create and run workflow
    workflow = SalesQualificationWorkflow()
    results = await workflow.process_company(company_name)
    
    # Display results
    print("\n📊 Sales Qualification Results")
    print("=" * 50)
    
    total_time = 0
    for step, result in results.items():
        print(f"\n### {step}")
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print("\n💭 Reasoning:")
            print(result["reasoning"])
            print("\n🔄 Analysis:")
            print(result["response"])
            print(f"\n⏱️ Time: {result['time']}s")
            total_time += result["time"]
    
    print(f"\nTotal Analysis Time: {total_time}s")

if __name__ == "__main__":
    asyncio.run(main()) 
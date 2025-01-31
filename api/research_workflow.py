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

class LiteratureAgent(Agent):
    """Agent specialized in literature review and current research"""
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER, logger: Optional[Callable] = None):
        self.logger = logger or print
        super().__init__(
            name="Literature Reviewer",
            role="Research Literature Analyst",
            goal="Review and analyze current research literature and developments",
            backstory="Expert in analyzing scientific literature and research papers",
            tools=[internet_search_tool],
            llm=ProxyAPITool(api_type=api_type, logger=logger)
        )

class GapAnalysisAgent(Agent):
    """Agent specialized in identifying research gaps and opportunities"""
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER, logger: Optional[Callable] = None):
        self.logger = logger or print
        super().__init__(
            name="Gap Analyzer",
            role="Research Gap Analyst",
            goal="Identify gaps, opportunities, and unexplored areas in research",
            backstory="Expert in identifying research opportunities and potential breakthroughs",
            tools=[internet_search_tool],
            llm=ProxyAPITool(logger=logger)
        )

class MethodologyAgent(Agent):
    """Agent specialized in research methodology and experimental design"""
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER, logger: Optional[Callable] = None):
        self.logger = logger or print
        super().__init__(
            name="Methodology Expert",
            role="Research Methodology Analyst",
            goal="Design and evaluate research methodologies and approaches",
            backstory="Expert in research design and experimental methodology",
            tools=[internet_search_tool],
            llm=ProxyAPITool(logger=logger)
        )

class ImpactAgent(Agent):
    """Agent specialized in impact assessment and future implications"""
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER, logger: Optional[Callable] = None):
        self.logger = logger or print
        super().__init__(
            name="Impact Assessor",
            role="Research Impact Analyst",
            goal="Evaluate potential impacts and future implications of research",
            backstory="Expert in assessing research impact and future developments",
            tools=[internet_search_tool],
            llm=ProxyAPITool(logger=logger)
        )

class SynthesisAgent(Agent):
    """Agent specialized in synthesizing research findings into a final report"""
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER, logger: Optional[Callable] = None):
        self.logger = logger or print
        super().__init__(
            name="Research Synthesizer",
            role="Research Report Writer",
            goal="Create a comprehensive, well-structured final research report",
            backstory="Expert in research synthesis and technical writing",
            tools=[internet_search_tool],
            llm=ProxyAPITool(logger=logger)
        )

class ResearchWorkflow:
    """Orchestrates the multi-agent research workflow"""
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER, logger: Optional[Callable] = None):
        self.logger = logger or print
        self.api_type = api_type
        
        # Initialize agents with logger and API type
        self.literature_agent = LiteratureAgent(api_type=api_type, logger=logger)
        self.gap_agent = GapAnalysisAgent(api_type=api_type, logger=logger)
        self.methodology_agent = MethodologyAgent(api_type=api_type, logger=logger)
        self.impact_agent = ImpactAgent(api_type=api_type, logger=logger)
        self.synthesis_agent = SynthesisAgent(api_type=api_type, logger=logger)
        
        # Create tasks for each agent
        self.tasks = [
            Task(
                description="Review current literature and research",
                expected_output="Comprehensive literature review",
                agent=self.literature_agent
            ),
            Task(
                description="Analyze research gaps and opportunities",
                expected_output="Gap analysis and opportunity identification",
                agent=self.gap_agent
            ),
            Task(
                description="Evaluate research methodologies",
                expected_output="Methodology assessment and recommendations",
                agent=self.methodology_agent
            ),
            Task(
                description="Assess research impact and implications",
                expected_output="Impact analysis and future predictions",
                agent=self.impact_agent
            ),
            Task(
                description="Synthesize final research report",
                expected_output="Comprehensive final report",
                agent=self.synthesis_agent
            )
        ]
    
    async def process_topic(self, topic: str) -> Dict:
        """Process a research topic through the multi-agent workflow"""
        self.logger(f"\n🔍 Starting research workflow for: {topic}")
        self.logger("=" * 50)
        
        results = {}
        context = ""
        
        for i, task in enumerate(self.tasks, 1):
            self.logger(f"\n📋 Step {i}: {task.description}")
            self.logger(f"🤖 Agent: {task.agent.name}")
            
            # Special handling for synthesis task
            if task.agent == self.synthesis_agent:
                prompt = f"""Research Topic: {topic}

Previous Analyses:
{context}

Your task is to synthesize all previous analyses into a clear, comprehensive final report.
Follow this structure:

EXECUTIVE SUMMARY
- Brief overview of key findings
- Major implications
- Critical recommendations

COMPREHENSIVE RESEARCH REPORT
1. Introduction
   - Research context
   - Objectives
   - Scope

2. Literature Review Summary
   - Current state of knowledge
   - Key theories and concepts
   - Recent developments

3. Research Gaps and Opportunities
   - Identified gaps
   - Emerging opportunities
   - Priority areas

4. Methodology Assessment
   - Research approaches
   - Data collection methods
   - Analysis techniques

5. Impact Analysis
   - Industry implications
   - Societal impact
   - Future predictions

6. Recommendations
   - Strategic priorities
   - Action items
   - Implementation considerations

7. Conclusion
   - Summary of findings
   - Future research directions
   - Final thoughts

Format the report professionally with clear sections, subsections, and bullet points where appropriate.
Focus on clarity, actionability, and practical implications."""
            else:
                prompt = f"""Research Topic: {topic}

Previous Analysis:
{context}

Your Task: {task.description}
As a {task.agent.role}, analyze this topic and provide:
1. Detailed analysis based on your expertise
2. Key findings and insights
3. Recommendations for next steps

Be thorough and analytical in your response."""

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
    # Get research topic
    topic = input("\n🎯 Enter research topic: ")
    
    # Create and run workflow
    workflow = ResearchWorkflow()
    results = await workflow.process_topic(topic)
    
    # Display results
    print("\n📊 Final Results")
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
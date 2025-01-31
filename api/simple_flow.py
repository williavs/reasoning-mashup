from praisonaiagents import Agent
import asyncio
import requests
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from typing import List, Dict
from .proxy_config import ProxyAPIConfig, ProxyAPITool

# Load environment variables
load_dotenv()

def internet_search_tool(query: str) -> List[Dict]:
    """
    Perform Internet Search using DuckDuckGo
    
    Args:
        query (str): The search query string
        
    Returns:
        List[Dict]: List of search results containing title, URL, and snippet
    """
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

class ResearchAgent(Agent):
    def __init__(self, api_type: str = ProxyAPIConfig.OPENROUTER):
        super().__init__(
            name="Research Expert",
            role="Scientific Research Analyst",
            goal="Analyze research topics comprehensively using both AI reasoning and internet research",
            backstory="You are an expert research analyst with deep knowledge across multiple scientific domains and the ability to gather real-time information",
            tools=[internet_search_tool],  # Use the tool function directly
            llm=ProxyAPITool(api_type=api_type)  # Use our proxy with specified API type
        )
    
    async def process_topic(self, topic: str) -> dict:
        """Process a research topic through our agent"""
        print(f"\n🔍 Analyzing topic: {topic}")
        print("=" * 50)
        
        # First, gather internet research using the tool
        print("\n🌐 Gathering internet research...")
        search_results = self.tools[0](topic)  # Call the tool function directly
        
        if not search_results:
            print("⚠️ No search results found. Proceeding with analysis using only AI knowledge.")
            search_context = "No internet research results available."
        else:
            print(f"✅ Found {len(search_results)} search results")
            search_context = "\n".join([
                f"Source: {result['title']}\nURL: {result['url']}\nSummary: {result['snippet']}\n"
                for result in search_results
            ])
        
        # Construct research prompt with search results
        prompt = f"""Analyze this research topic: {topic}

Available internet research:
{search_context}

Based on both the internet research and your knowledge, please provide:
1. Current state of research
2. Key challenges and opportunities
3. Potential future directions
4. Practical applications

Be thorough but comprehensive, and incorporate relevant findings from the internet research. your output should read like a report"""
        
        # Process through our proxy API
        result = await self.llm.execute(prompt)
        
        if result:
            print("✅ Analysis complete")
            print(f"⏱️  Time: {result.get('elapsed_time', 'N/A')}s")
            print("\n💭 Reasoning:")
            print(result.get("reasoning", "No reasoning provided"))
            print("\n🔄 Response:")
            print(result.get("response", "No response provided"))
            return result
        else:
            print("❌ Analysis failed")
            return None

async def main():
    # Get research topic
    topic = input("\n🎯 Enter research topic: ")
    
    # Create and run agent
    agent = ResearchAgent()
    result = await agent.process_topic(topic)
    
    if result:
        print("\n📊 Final Results")
        print("=" * 50)
        print("\n💡 Key Insights:")
        print(result.get("response", "No results available"))
    else:
        print("\n❌ Analysis failed to complete")

if __name__ == "__main__":
    asyncio.run(main())
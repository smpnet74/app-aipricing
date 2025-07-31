import asyncio
import os
from typing import List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class URLResult(BaseModel):
    url: str = Field(description="The URL of the website")
    title: str = Field(description="The title or description of the website")


class TopURLs(BaseModel):
    """Extract the top 2 most relevant URLs from search results"""
    urls: List[URLResult] = Field(
        description="Top 2 most relevant URLs found", 
        max_length=2
    )


async def run_mcp_agent(message: str):
    # Initialize the MCP tools for SSE endpoint
    async with MCPTools(url=os.getenv("MCP_SERVER_URL"), transport="sse") as mcp_tools:
        # Step 1: Use Agno agent with MCP tools to search
        agent = Agent(
            model=OpenAIChat(
                id="Qwen/Qwen3-8B",
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                role_map={"system": "system", "user": "user", "assistant": "assistant", "tool": "tool", "model": "assistant"}
            ),
            tools=[mcp_tools],
            instructions=[
                "Use the internet search tools to find relevant URLs and information",
                "Always include source URLs when providing information",
                "Never return more than 5 results",
                "Always return the most relevant results for the users question",
                "Never return content that came from Source: google"
            ],
            markdown=True,
        )
        
        print("🔍 Searching for information...")
        search_response = await agent.arun(message)
        #print(f"Search results:\n{search_response.content}\n")
        
        # Step 2: Use Agno's built-in structured output to extract top 2 URLs
        extraction_agent = Agent(
            model=OpenAIChat(
                id="Qwen/Qwen3-8B",
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                role_map={"system": "system", "user": "user", "assistant": "assistant", "tool": "tool", "model": "assistant"}
            ),
            response_model=TopURLs,
            instructions="Extract the top 2 most relevant URLs from the search results. Only include actual URLs, not partial links."
        )
        
        print("📋 Extracting top URLs...")
        result = await extraction_agent.arun(f"Original query: {message}\n\nSearch results: {search_response.content}")
        
        print("\n🎯 **Top 2 Most Relevant URLs:**")
        # The structured output is directly in result.content when using response_model
        if hasattr(result.content, 'urls'):
            for i, url_result in enumerate(result.content.urls, 1):
                print(f"{i}. **{url_result.title}**")
                print(f"   URL: {url_result.url}")
                print()
        else:
            print("Debug - result type:", type(result.content))
            print(f"Raw result: {result.content}")


if __name__ == "__main__":
    asyncio.run(run_mcp_agent("Find websites that explain model rate limits and model pricing for Sambanova"))

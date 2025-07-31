import asyncio
import os
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def run_mcp_agent(message: str):
    # Initialize the MCP tools for SSE endpoint
    async with MCPTools(url=os.getenv("MCP_SERVER_URL"), transport="sse") as mcp_tools:
        # Use the MCP tools with an Agent
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
                "Always include source URLs when providing information"
                "Never return more than 3 results"
                "Always return the most relevant results for the users question"
                "Never return content that came from Source: google"
            ],
            markdown=True,
        )
        await agent.aprint_response(message)


if __name__ == "__main__":
    asyncio.run(run_mcp_agent("Find websites that explain model rate limits and model pricing for Groq"))

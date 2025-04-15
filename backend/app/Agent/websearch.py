import asyncio
import nest_asyncio
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Apply nest_asyncio to solve event loop conflicts
nest_asyncio.apply()

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

# Initialize DuckDuckGo search with optimized configuration
search_tool = DuckDuckGoSearchResults(
    max_results=5,  # Limit number of results
    backend="api",  # Use API backend for better results
    results_separator="\n\n",  # Better separation between results
    return_direct=False,  # Don't return results directly to allow for processing
)

# Use a lazy-loading pattern for browser tools
_async_browser = None
_browser_tools = None

async def get_browser():
    """Lazy initialization of browser instance"""
    global _async_browser
    if _async_browser is None:
        try:
            _async_browser = await create_async_playwright_browser()
        except Exception as e:
            print(f"Error initializing browser: {e}")
            return None
    return _async_browser

async def get_browser_tools():
    """Lazy initialization of browser tools"""
    global _browser_tools
    if _browser_tools is None:
        browser = await get_browser()
        if browser:
            toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
            _browser_tools = toolkit.get_tools()
        else:
            _browser_tools = []
    return _browser_tools

class WebSearchRequest(BaseModel):
    """Web search request parameters."""
    query: str = Field(..., description="The search query to execute")
    num_results: int = Field(default=5, description="Number of search results to return")

class BrowserNavigateRequest(BaseModel):
    """Browser navigation request parameters."""
    url: str = Field(..., description="URL to navigate to")

@tool
async def web_search(query: str) -> str:
    """
    Search the web for information using DuckDuckGo.
    
    Args:
        query: The search query to execute
        
    Returns:
        Search results from DuckDuckGo
    """
    try:
        # Use ainvoke for async operation
        results = await search_tool.ainvoke(query)
        
        # Process and format results
        if isinstance(results, str):
            # Clean up and format the results
            cleaned_results = results.strip()
            # Remove any duplicate newlines
            cleaned_results = "\n".join(line for line in cleaned_results.splitlines() if line.strip())
            return cleaned_results
        return str(results)
    except Exception as e:
        print(f"Error in web_search: {str(e)}")
        return f"Error performing web search: {str(e)}"

@tool
async def browse_website(url: str) -> str:
    """
    Browse a website and extract its content using Playwright.
    
    Args:
        url: URL to navigate to and extract content from
        
    Returns:
        The extracted content from the website
    """
    try:
        browser_tools = await get_browser_tools()
        if not browser_tools:
            return f"Error: Browser tools not available. Cannot browse {url}"
        
        get_content_tool = next(
            (tool for tool in browser_tools if tool.name == "get_page_content"), 
            None
        )
        
        if not get_content_tool:
            return "Error: Page content extraction tool not available"
        
        navigate_tool = next(
            (tool for tool in browser_tools if tool.name == "navigate_browser"), 
            None
        )
        
        if navigate_tool:
            await navigate_tool.ainvoke({"url": url})
            
        content = await get_content_tool.ainvoke({})
        
        # Clean up and format the content
        if content:
            # Remove excessive whitespace and format
            content = "\n".join(line for line in content.splitlines() if line.strip())
            return content
        return "No content extracted from the webpage"
    except Exception as e:
        print(f"Error in browse_website: {str(e)}")
        return f"Error browsing website: {str(e)}"

async def search_web(query: str) -> str:
    """
    Search the web and optionally browse the top result.
    
    Args:
        query: The search query to execute
        
    Returns:
        Combined search results and optionally page content
    """
    try:
        search_results = await web_search(query)
        
        # Parse the results to extract the first URL
        lines = search_results.split('\n')
        urls = [line for line in lines if line.startswith('http')]
        
        if urls:
            first_url = urls[0].strip()
            page_content = await browse_website(first_url)
            
            # Format the combined results
            combined_results = (
                "Search Results:\n"
                f"{search_results}\n\n"
                "Detailed Content from Top Result:\n"
                f"{page_content}"
            )
            return combined_results
        
        return search_results
    except Exception as e:
        print(f"Error in search_web: {str(e)}")
        return f"Error performing web search and browsing: {str(e)}"

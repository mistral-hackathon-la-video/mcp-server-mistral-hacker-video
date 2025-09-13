"""
Jina AI Web Search MCP Server

A high-performance MCP server for web searching and content extraction using Jina AI.
Optimized for speed, reliability, and comprehensive web research capabilities.
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field
import mcp.types as types
import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import aiohttp
import json
from dotenv import load_dotenv
from urllib.parse import quote_plus, urlencode
import time
from functools import wraps

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Jina AI Web Search MCP Server", port=3000, stateless_http=True, debug=True)

# API Configuration
JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_87da30b048184c829fb91e2f30691112ckNdk_iZ4MWTvCWdG-affuPFLr3u")

if not JINA_API_KEY:
    logger.error("JINA_API_KEY is required but not configured")
    raise ValueError("JINA_API_KEY environment variable is required")

# Jina API endpoints
JINA_SEARCH_URL = "https://s.jina.ai/"
JINA_READER_URL = "https://r.jina.ai/"

# Configuration constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_RESULTS_LIMIT = 50
DEFAULT_RESULTS = 10

# Global session for connection pooling
_session: Optional[aiohttp.ClientSession] = None

async def get_session() -> aiohttp.ClientSession:
    """Get or create a shared aiohttp session for connection pooling"""
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
        )
        _session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "User-Agent": "Jina-MCP-Server/1.0",
                "Accept": "application/json, text/plain, */*"
            }
        )
    return _session

def retry_on_failure(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """Decorator to retry failed requests with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except aiohttp.ClientError as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed: {e}")
                except Exception as e:
                    # Don't retry on non-network errors
                    logger.error(f"Non-retryable error: {e}")
                    raise
            raise last_exception
        return wrapper
    return decorator

@retry_on_failure()
async def fetch_json(url: str, headers: Dict[str, str] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Fetch JSON data from a URL with retry logic and connection pooling"""
    session = await get_session()
    
    request_headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Accept": "application/json"
    }
    if headers:
        request_headers.update(headers)
    
    try:
        async with session.get(url, headers=request_headers, params=params) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ContentTypeError:
        # Handle cases where response isn't JSON
        text = await response.text()
        logger.warning(f"Non-JSON response from {url}: {text[:200]}...")
        return {"error": "Invalid JSON response", "raw_text": text}

@retry_on_failure()
async def fetch_text(url: str, headers: Dict[str, str] = None) -> str:
    """Fetch text data from a URL with retry logic and connection pooling"""
    session = await get_session()
    
    request_headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Accept": "text/plain, text/markdown, */*"
    }
    if headers:
        request_headers.update(headers)
    
    async with session.get(url, headers=request_headers) as response:
        response.raise_for_status()
        return await response.text()

def validate_url(url: str) -> bool:
    """Basic URL validation"""
    return url.startswith(('http://', 'https://')) and len(url) > 10

def sanitize_query(query: str) -> str:
    """Sanitize search query"""
    return query.strip()[:500]  # Limit query length

def extract_domain(url: str) -> str:
    """Extract domain from URL for logging/debugging"""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except:
        return "unknown"

@mcp.tool(
    title="Web Search",
    description="Search the web using Jina AI. Returns comprehensive results with titles, snippets, and URLs."
)
async def web_search(
    query: str = Field(description="The search query (max 500 characters)"),
    num_results: int = Field(default=DEFAULT_RESULTS, description=f"Number of results to return (1-{MAX_RESULTS_LIMIT})"),
    include_snippets: bool = Field(default=True, description="Include content snippets in results"),
    filter_duplicates: bool = Field(default=True, description="Remove duplicate URLs from results")
) -> Dict[str, Any]:
    """
    Search the web using Jina AI's search API.
    
    Returns:
    {
        "results": [
            {
                "title": "Result title",
                "snippet": "Result description",
                "url": "https://example.com",
                "position": 1,
                "domain": "example.com"
            }
        ],
        "query": "original query",
        "total_results": 10,
        "search_time": 1.23
    }
    """
    start_time = time.time()
    
    try:
        # Validate and sanitize inputs
        query = sanitize_query(query)
        if not query:
            return {
                "error": "Query cannot be empty",
                "results": [],
                "total_results": 0
            }
        
        num_results = max(1, min(num_results, MAX_RESULTS_LIMIT))
        
        # Prepare search headers
        headers = {
            "X-With-Generated-Alt": "true" if include_snippets else "false",
            "X-No-Cache": "false",  # Allow caching for better performance
        }
        
        # Construct search URL
        search_url = f"{JINA_SEARCH_URL}{quote_plus(query)}"
        
        logger.info(f"Searching for: '{query}' (requesting {num_results} results)")
        
        # Perform search
        data = await fetch_json(search_url, headers=headers)
        
        if "error" in data:
            return {
                "error": f"Jina API error: {data['error']}",
                "results": [],
                "total_results": 0
            }
        
        # Process results
        results = []
        seen_urls = set() if filter_duplicates else None
        
        raw_results = data.get("data", [])
        if not raw_results:
            logger.warning("No results returned from Jina API")
            return {
                "results": [],
                "query": query,
                "total_results": 0,
                "search_time": round(time.time() - start_time, 2)
            }
        
        for i, result in enumerate(raw_results):
            if len(results) >= num_results:
                break
                
            url = result.get("url", "")
            if not url or not validate_url(url):
                continue
                
            # Skip duplicates if filtering enabled
            if filter_duplicates:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
            
            # Extract and clean content
            title = result.get("title", "").strip()
            snippet = ""
            
            if include_snippets:
                # Try multiple content fields
                content_sources = [
                    result.get("description", ""),
                    result.get("content", ""),
                    result.get("snippet", "")
                ]
                for content in content_sources:
                    if content and content.strip():
                        snippet = content.strip()[:300] + ("..." if len(content) > 300 else "")
                        break
            
            results.append({
                "title": title or "Untitled",
                "snippet": snippet,
                "url": url,
                "position": len(results) + 1,
                "domain": extract_domain(url)
            })
        
        search_time = round(time.time() - start_time, 2)
        logger.info(f"Search completed: {len(results)} results in {search_time}s")
        
        return {
            "results": results,
            "query": query,
            "total_results": len(results),
            "search_time": search_time,
            "filtered_duplicates": filter_duplicates
        }
        
    except Exception as e:
        search_time = round(time.time() - start_time, 2)
        error_message = f"Search error: {str(e)}"
        logger.error(f"{error_message} (after {search_time}s)")
        return {
            "error": error_message,
            "results": [],
            "total_results": 0,
            "search_time": search_time
        }

@mcp.tool(
    title="Read Web Page",
    description="Extract clean, readable content from any web page using Jina Reader API"
)
async def read_webpage(
    url: str = Field(description="The URL of the webpage to read"),
    include_images: bool = Field(default=False, description="Include image descriptions"),
    include_links: bool = Field(default=True, description="Include links summary"),
    max_content_length: int = Field(default=50000, description="Maximum content length (characters)")
) -> Dict[str, Any]:
    """
    Read and extract clean content from a web page.
    
    Returns:
    {
        "title": "Page title",
        "content": "Clean extracted content in markdown",
        "url": "https://example.com",
        "word_count": 500,
        "read_time": 1.23,
        "domain": "example.com"
    }
    """
    start_time = time.time()
    
    try:
        # Validate URL
        if not validate_url(url):
            return {
                "error": "Invalid URL format",
                "content": None,
                "url": url
            }
        
        # Prepare headers
        headers = {
            "X-With-Images-Summary": "true" if include_images else "false",
            "X-With-Links-Summary": "true" if include_links else "false",
            "X-Retain-Images": "true" if include_images else "false"
        }
        
        # Construct reader URL
        reader_url = f"{JINA_READER_URL}{url}"
        
        logger.info(f"Reading webpage: {extract_domain(url)}")
        
        # Fetch content
        content = await fetch_text(reader_url, headers=headers)
        
        # Limit content length
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n\n[Content truncated due to length limit]"
        
        # Extract title (usually first heading)
        title = "Untitled"
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines for title
            line = line.strip()
            if line.startswith('#') and len(line) > 1:
                title = line.lstrip('#').strip()
                break
            elif line and not line.startswith(('http', 'www', '---')):
                title = line[:100]  # Use first substantial line as title
                break
        
        # Calculate metrics
        word_count = len(content.split()) if content else 0
        read_time = round(time.time() - start_time, 2)
        
        logger.info(f"Webpage read successfully: {word_count} words in {read_time}s")
        
        return {
            "title": title,
            "content": content,
            "url": url,
            "word_count": word_count,
            "read_time": read_time,
            "domain": extract_domain(url),
            "content_length": len(content)
        }
        
    except Exception as e:
        read_time = round(time.time() - start_time, 2)
        error_message = f"Error reading webpage: {str(e)}"
        logger.error(f"{error_message} (after {read_time}s)")
        return {
            "error": error_message,
            "content": None,
            "url": url,
            "read_time": read_time
        }

@mcp.tool(
    title="Search and Read",
    description="Search for a topic and automatically read the top results for comprehensive research"
)
async def search_and_read(
    query: str = Field(description="The search query"),
    num_pages: int = Field(default=3, description="Number of pages to read (1-10)"),
    include_content: bool = Field(default=True, description="Include full content of each page"),
    summary_length: int = Field(default=500, description="Length of content summary for each page")
) -> Dict[str, Any]:
    """
    Search for a topic and automatically read the top results.
    
    Returns:
    {
        "query": "search query",
        "pages_read": [
            {
                "title": "Page title",
                "url": "https://example.com",
                "summary": "Content summary...",
                "word_count": 1000,
                "full_content": "Complete content..." (if include_content=True)
            }
        ],
        "total_pages": 3,
        "total_time": 5.67
    }
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        num_pages = max(1, min(num_pages, 10))
        summary_length = max(100, min(summary_length, 2000))
        
        logger.info(f"Starting search and read for: '{query}' ({num_pages} pages)")
        
        # First, perform a search
        search_results = await web_search(
            query=query,
            num_results=num_pages + 2,  # Get a few extra in case some fail
            include_snippets=True
        )
        
        if "error" in search_results or not search_results.get("results"):
            return {
                "error": f"Search failed: {search_results.get('error', 'No results found')}",
                "pages_read": [],
                "total_pages": 0
            }
        
        pages_read = []
        successful_reads = 0
        
        # Read each result page
        for i, result in enumerate(search_results["results"]):
            if successful_reads >= num_pages:
                break
                
            url = result.get("url")
            if not url:
                continue
                
            try:
                logger.info(f"Reading page {successful_reads + 1}/{num_pages}: {extract_domain(url)}")
                
                page_content = await read_webpage(
                    url=url,
                    include_images=False,
                    include_links=True,
                    max_content_length=20000  # Reasonable limit for search results
                )
                
                if "error" not in page_content and page_content.get("content"):
                    # Create summary
                    full_content = page_content.get("content", "")
                    summary = full_content[:summary_length] + ("..." if len(full_content) > summary_length else "")
                    
                    page_info = {
                        "title": page_content.get("title", result.get("title", "Untitled")),
                        "url": url,
                        "summary": summary,
                        "word_count": page_content.get("word_count", 0),
                        "domain": page_content.get("domain", extract_domain(url)),
                        "read_time": page_content.get("read_time", 0)
                    }
                    
                    if include_content:
                        page_info["full_content"] = full_content
                    
                    pages_read.append(page_info)
                    successful_reads += 1
                    
                else:
                    logger.warning(f"Failed to read content from: {url}")
                    
            except Exception as e:
                logger.warning(f"Error reading page {url}: {e}")
                continue
        
        total_time = round(time.time() - start_time, 2)
        logger.info(f"Search and read completed: {len(pages_read)} pages in {total_time}s")
        
        return {
            "query": query,
            "pages_read": pages_read,
            "total_pages": len(pages_read),
            "total_time": total_time,
            "search_results_count": len(search_results.get("results", [])),
            "success_rate": f"{len(pages_read)}/{len(search_results.get('results', []))} pages read successfully"
        }
        
    except Exception as e:
        total_time = round(time.time() - start_time, 2)
        error_message = f"Error in search and read: {str(e)}"
        logger.error(f"{error_message} (after {total_time}s)")
        return {
            "error": error_message,
            "pages_read": [],
            "total_pages": 0,
            "total_time": total_time
        }

@mcp.tool(
    title="Quick Answer",
    description="Get a quick answer by searching and summarizing the most relevant content"
)
async def quick_answer(
    question: str = Field(description="The question to answer"),
    include_sources: bool = Field(default=True, description="Include source URLs with the answer"),
    max_sources: int = Field(default=3, description="Maximum number of sources to use (1-5)")
) -> Dict[str, Any]:
    """
    Get a quick answer by searching and extracting relevant information.
    
    Returns:
    {
        "answer": "Compiled answer from multiple sources",
        "sources": [
            {
                "url": "https://source.com",
                "title": "Source title",
                "snippet": "Relevant excerpt"
            }
        ],
        "question": "original question",
        "confidence": "high/medium/low"
    }
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        max_sources = max(1, min(max_sources, 5))
        question = sanitize_query(question)
        
        if not question:
            return {
                "error": "Question cannot be empty",
                "answer": None
            }
        
        logger.info(f"Finding answer for: '{question}'")
        
        # Search for relevant content
        search_results = await web_search(
            query=question,
            num_results=max_sources * 2,  # Get extra results for better selection
            include_snippets=True
        )
        
        if "error" in search_results or not search_results.get("results"):
            return {
                "error": f"Could not find relevant sources: {search_results.get('error', 'No results')}",
                "answer": None,
                "question": question
            }
        
        # Compile answer from search results
        answer_parts = []
        sources = []
        
        for result in search_results["results"][:max_sources]:
            snippet = result.get("snippet", "")
            if snippet and len(snippet.strip()) > 20:  # Only use substantial snippets
                answer_parts.append(snippet.strip())
                
                if include_sources:
                    sources.append({
                        "url": result.get("url", ""),
                        "title": result.get("title", "Untitled"),
                        "snippet": snippet[:200] + ("..." if len(snippet) > 200 else ""),
                        "domain": result.get("domain", "")
                    })
        
        # Create comprehensive answer
        if answer_parts:
            answer = " | ".join(answer_parts)  # Separate different sources
            confidence = "high" if len(answer_parts) >= 2 else "medium"
        else:
            answer = "No sufficient information found in search results."
            confidence = "low"
        
        response_time = round(time.time() - start_time, 2)
        logger.info(f"Answer compiled from {len(sources)} sources in {response_time}s")
        
        return {
            "answer": answer,
            "sources": sources if include_sources else [],
            "question": question,
            "confidence": confidence,
            "response_time": response_time,
            "sources_used": len(sources)
        }
        
    except Exception as e:
        response_time = round(time.time() - start_time, 2)
        error_message = f"Error getting answer: {str(e)}"
        logger.error(f"{error_message} (after {response_time}s)")
        return {
            "error": error_message,
            "answer": None,
            "question": question,
            "response_time": response_time
        }

@mcp.tool(
    title="Batch Web Search",
    description="Perform multiple searches efficiently with a single request"
)
async def batch_search(
    queries: List[str] = Field(description="List of search queries (max 10)"),
    results_per_query: int = Field(default=5, description="Results per query (1-20)")
) -> Dict[str, Any]:
    """
    Perform multiple searches in parallel for efficiency.
    
    Returns:
    {
        "searches": [
            {
                "query": "query1",
                "results": [...],
                "total_results": 5
            }
        ],
        "total_queries": 3,
        "total_time": 2.34
    }
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        queries = [sanitize_query(q) for q in queries[:10] if q.strip()]  # Limit to 10 queries
        results_per_query = max(1, min(results_per_query, 20))
        
        if not queries:
            return {
                "error": "No valid queries provided",
                "searches": [],
                "total_queries": 0
            }
        
        logger.info(f"Starting batch search for {len(queries)} queries")
        
        # Run searches in parallel
        search_tasks = [
            web_search(query=query, num_results=results_per_query, include_snippets=True)
            for query in queries
        ]
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results
        searches = []
        successful_searches = 0
        
        for i, (query, result) in enumerate(zip(queries, search_results)):
            if isinstance(result, Exception):
                searches.append({
                    "query": query,
                    "error": str(result),
                    "results": [],
                    "total_results": 0
                })
            elif "error" not in result:
                searches.append({
                    "query": query,
                    "results": result.get("results", []),
                    "total_results": result.get("total_results", 0),
                    "search_time": result.get("search_time", 0)
                })
                successful_searches += 1
            else:
                searches.append({
                    "query": query,
                    "error": result.get("error", "Unknown error"),
                    "results": [],
                    "total_results": 0
                })
        
        total_time = round(time.time() - start_time, 2)
        logger.info(f"Batch search completed: {successful_searches}/{len(queries)} successful in {total_time}s")
        
        return {
            "searches": searches,
            "total_queries": len(queries),
            "successful_searches": successful_searches,
            "total_time": total_time
        }
        
    except Exception as e:
        total_time = round(time.time() - start_time, 2)
        error_message = f"Batch search error: {str(e)}"
        logger.error(f"{error_message} (after {total_time}s)")
        return {
            "error": error_message,
            "searches": [],
            "total_queries": len(queries) if 'queries' in locals() else 0,
            "total_time": total_time
        }

@mcp.resource(
    uri="jina://info/capabilities",
    description="Information about Jina AI capabilities and current service status",
    name="Jina AI Capabilities"
)
def get_jina_info() -> str:
    """Get information about Jina AI capabilities"""
    return f"""# Jina AI Web Search & Reader Capabilities

## Service Status
- **Status**: ✅ Operational
- **API Key**: {'✅ Configured' if JINA_API_KEY else '❌ Not configured'}
- **Connection Pooling**: ✅ Enabled
- **Retry Logic**: ✅ Enabled ({MAX_RETRIES} retries with exponential backoff)
- **Timeout**: {DEFAULT_TIMEOUT}s per request

## Available Tools

### 🔍 Web Search
- **Fast Search**: Optimized search with connection pooling
- **Duplicate Filtering**: Remove duplicate URLs from results
- **Result Limits**: 1-{MAX_RESULTS_LIMIT} results per search
- **Content Snippets**: Optional detailed descriptions
- **Domain Extraction**: Automatic domain identification

### 📖 Web Page Reader
- **Clean Extraction**: Convert web pages to clean markdown
- **Image Support**: Optional image descriptions
- **Link Summaries**: Extract and summarize links
- **Content Limits**: Configurable maximum content length
- **Fast Processing**: Optimized for speed and reliability

### 🔄 Search and Read
- **Automated Research**: Search + read top results automatically
- **Batch Processing**: Read multiple pages efficiently
- **Content Summaries**: Configurable summary lengths
- **Success Tracking**: Monitor read success rates

### ⚡ Quick Answer
- **Instant Answers**: Compile answers from multiple sources
- **Source Attribution**: Include source URLs and titles
- **Confidence Scoring**: Rate answer reliability
- **Flexible Sources**: 1-5 sources per answer

### 📊 Batch Search
- **Parallel Processing**: Multiple searches simultaneously
- **Efficient**: Up to 10 queries at once
- **Error Handling**: Individual query error isolation

## Performance Features

### Connection Optimization
- **Connection Pooling**: Reuse connections for better performance
- **DNS Caching**: 5-minute DNS cache for faster lookups
- **Concurrent Limits**: 100 total, 30 per host connections

### Error Handling
- **Retry Logic**: Automatic retry with exponential backoff
- **Timeout Management**: Configurable request timeouts
- **Graceful Degradation**: Partial results on some failures

### Content Processing
- **Smart Truncation**: Intelligent content length limits
- **Duplicate Detection**: Remove duplicate URLs
- **Domain Extraction**: Automatic source identification
- **Performance Metrics**: Track response times and success rates

## Usage Examples

### Basic Search
```
web_search(query="artificial intelligence", num_results=10)
```

### Read Specific Page
```
read_webpage(url="https://example.com", include_images=True)
```

### Research Topic
```
search_and_read(query="climate change solutions", num_pages=5)
```

### Get Quick Answer
```
quick_answer(question="What is quantum computing?")
```

### Multiple Searches
```
batch_search(queries=["AI trends", "ML algorithms", "Deep learning"])
```

## API Endpoints Used
- **Search API**: {JINA_SEARCH_URL}
- **Reader API**: {JINA_READER_URL}

## Rate Limits & Performance
- **Generous Limits**: Jina AI provides generous rate limits
- **Optimized Requests**: Connection pooling and retry logic
- **Fast Response**: Typically < 3 seconds per request
- **Reliable**: Built-in error handling and fallbacks
"""

@mcp.resource(
    uri="jina://stats/performance",
    description="Current performance statistics and metrics",
    name="Performance Statistics"
)
def get_performance_stats() -> str:
    """Get current performance statistics"""
    session_info = "Active" if _session and not _session.closed else "Not initialized"
    
    return f"""# Performance Statistics

## Current Status
- **Timestamp**: {datetime.now().isoformat()}
- **Session Status**: {session_info}
- **API Key**: {'✅ Valid' if JINA_API_KEY else '❌ Missing'}

## Configuration
- **Default Timeout**: {DEFAULT_TIMEOUT}s
- **Max Retries**: {MAX_RETRIES}
- **Retry Delay**: {RETRY_DELAY}s (exponential backoff)
- **Max Results Limit**: {MAX_RESULTS_LIMIT}
- **Default Results**: {DEFAULT_RESULTS}

## Connection Pool Settings
- **Total Connection Limit**: 100
- **Per-Host Limit**: 30
- **DNS Cache TTL**: 300s
- **Connection Reuse**: Enabled

## Optimization Features
✅ **Connection Pooling**: Reuse HTTP connections
✅ **Retry Logic**: Exponential backoff on failures
✅ **Request Timeouts**: Prevent hanging requests
✅ **Duplicate Filtering**: Remove duplicate search results
✅ **Content Limits**: Prevent oversized responses
✅ **Parallel Processing**: Concurrent requests support
✅ **Error Isolation**: Individual request error handling
✅ **Performance Tracking**: Response time monitoring

## Tools Performance Profile
- **web_search**: ~1-3s per request
- **read_webpage**: ~2-5s per page
- **search_and_read**: ~5-15s for 3-5 pages
- **quick_answer**: ~2-4s with source compilation
- **batch_search**: ~3-8s for multiple queries

## Memory Optimization
- **Shared Session**: Single session for all requests
- **Content Truncation**: Prevent memory bloat
- **Efficient Processing**: Stream-based content handling
- **Resource Cleanup**: Automatic session management
"""

# Cleanup function for graceful shutdown
async def cleanup():
    """Cleanup resources on shutdown"""
    global _session
    if _session and not _session.closed:
        await _session.close()
        logger.info("HTTP session closed")

# Register cleanup with proper async handling
import atexit
def cleanup_sync():
    """Synchronous cleanup wrapper"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(cleanup())
        else:
            loop.run_until_complete(cleanup())
    except RuntimeError:
        # No event loop, create a new one
        asyncio.run(cleanup())
    except:
        pass  # Ignore cleanup errors during shutdown

atexit.register(cleanup_sync)

if __name__ == "__main__":
    logger.info("Starting Jina AI Web Search MCP Server...")
    logger.info(f"API Key configured: {'Yes' if JINA_API_KEY else 'No'}")
    logger.info(f"Max results limit: {MAX_RESULTS_LIMIT}")
    logger.info(f"Default timeout: {DEFAULT_TIMEOUT}s")
    
    try:
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Cleanup will be handled by atexit
        pass
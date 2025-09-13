# Jina AI Web Search MCP Server

A high-performance MCP (Model Context Protocol) server optimized for web searching and content extraction using Jina AI. Built for speed, reliability, and comprehensive web research capabilities.

## 🚀 Key Features

### ⚡ Performance Optimized
- **Connection Pooling**: Reuses HTTP connections for 3x faster requests
- **Retry Logic**: Exponential backoff on failures with automatic recovery
- **Parallel Processing**: Concurrent requests for batch operations
- **Smart Caching**: DNS caching and request optimization
- **Resource Management**: Automatic cleanup and memory optimization

### 🔍 Comprehensive Search
- **Web Search**: Fast, accurate search with Jina AI
- **Content Reading**: Extract clean markdown from any webpage
- **Batch Search**: Multiple queries processed in parallel
- **Smart Filtering**: Duplicate removal and content validation
- **Research Mode**: Automated search + read for deep research

### 🛡️ Robust & Reliable
- **Error Handling**: Graceful degradation with detailed error reporting
- **Timeout Management**: Configurable timeouts prevent hanging
- **Input Validation**: Sanitization and bounds checking
- **Monitoring**: Performance tracking and success metrics

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- UV package manager

### Quick Start

1. **Clone and setup**:
```bash
git clone <repository-url>
cd jina-web-search-mcp
export PATH="$HOME/.local/bin:$PATH"  # Add UV to PATH
```

2. **Install dependencies**:
```bash
uv lock --upgrade
uv sync
```

3. **Start the server**:
```bash
uv run main.py
```

The server will start on port 3000 with Jina AI pre-configured.

## 🔧 Configuration

### API Key
The Jina AI API key is pre-configured. To use your own:

```bash
# Create .env file
echo "JINA_API_KEY=your_jina_api_key_here" > .env
```

Get your API key at: https://jina.ai

### Performance Tuning
Configure these constants in `main.py`:

```python
DEFAULT_TIMEOUT = 30        # Request timeout (seconds)
MAX_RETRIES = 3            # Retry attempts
MAX_RESULTS_LIMIT = 50     # Maximum results per search
```

## 🛠️ Available Tools

### 1. Web Search
Fast, comprehensive web search with filtering and optimization.

```python
web_search(
    query="artificial intelligence trends 2024",
    num_results=10,
    include_snippets=True,
    filter_duplicates=True
)
```

**Returns**: Results with titles, snippets, URLs, and metadata
**Performance**: ~1-3 seconds per search

### 2. Read Web Page
Extract clean, readable content from any webpage.

```python
read_webpage(
    url="https://example.com/article",
    include_images=False,
    include_links=True,
    max_content_length=50000
)
```

**Returns**: Clean markdown content with metadata
**Performance**: ~2-5 seconds per page

### 3. Search and Read
Automated research combining search with content extraction.

```python
search_and_read(
    query="quantum computing breakthroughs",
    num_pages=5,
    include_content=True,
    summary_length=500
)
```

**Returns**: Comprehensive research with summaries and full content
**Performance**: ~5-15 seconds for 3-5 pages

### 4. Quick Answer
Get instant answers compiled from multiple sources.

```python
quick_answer(
    question="What is machine learning?",
    include_sources=True,
    max_sources=3
)
```

**Returns**: Compiled answer with source attribution
**Performance**: ~2-4 seconds with sources

### 5. Batch Search
Process multiple search queries in parallel.

```python
batch_search(
    queries=["AI trends", "ML algorithms", "Deep learning"],
    results_per_query=5
)
```

**Returns**: Results for all queries with performance metrics
**Performance**: ~3-8 seconds for multiple queries

## 🔗 Integration

### Claude Desktop
Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "jina-search": {
      "command": "uv",
      "args": ["run", "main.py"],
      "cwd": "/path/to/jina-web-search-mcp"
    }
  }
}
```

### MCP Inspector
Test your server interactively:

```bash
npx @modelcontextprotocol/inspector
```

Connect to: `http://127.0.0.1:3000/mcp`

## 📊 Resources

### Service Information
- `jina://info/capabilities` - Detailed capabilities and status
- `jina://stats/performance` - Performance metrics and configuration

## 💡 Usage Examples

### Research a Topic
```python
# Comprehensive research
results = await search_and_read(
    query="renewable energy innovations 2024",
    num_pages=5,
    include_content=True
)

# Process results
for page in results["pages_read"]:
    print(f"Title: {page['title']}")
    print(f"Summary: {page['summary']}")
    print(f"Full content available: {len(page.get('full_content', ''))}")
```

### Fact Checking
```python
# Get quick answer with sources
answer = await quick_answer(
    question="Is renewable energy cost-effective?",
    include_sources=True,
    max_sources=5
)

# Verify with detailed reading
for source in answer["sources"]:
    content = await read_webpage(
        url=source["url"],
        include_links=True
    )
    # Analyze content for verification
```

### Batch Research
```python
# Multiple related searches
topics = [
    "artificial intelligence ethics",
    "AI regulation 2024",
    "machine learning bias",
    "AI safety research"
]

results = await batch_search(
    queries=topics,
    results_per_query=8
)

# Process all results
for search in results["searches"]:
    print(f"Topic: {search['query']}")
    print(f"Results: {len(search['results'])}")
```

## ⚡ Performance Features

### Connection Optimization
- **Pool Size**: 100 total connections, 30 per host
- **DNS Caching**: 5-minute cache for faster lookups
- **Keep-Alive**: Connection reuse for multiple requests
- **Timeout Handling**: Prevents hanging requests

### Error Handling
- **Retry Logic**: 3 attempts with exponential backoff
- **Graceful Degradation**: Partial results on some failures
- **Error Isolation**: Individual request failures don't affect others
- **Detailed Logging**: Comprehensive error reporting

### Content Processing
- **Smart Truncation**: Intelligent content length limits
- **Duplicate Detection**: Remove duplicate URLs automatically
- **Input Validation**: Sanitize and validate all inputs
- **Memory Management**: Efficient content handling

## 🧪 Testing

Run the optimized server:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run main.py
```

The server includes built-in performance monitoring and will log:
- Request times
- Success rates
- Error details
- Connection pool status

## 🔍 Troubleshooting

### Common Issues

**Server won't start**:
```bash
# Check UV installation
uv --version

# Reinstall dependencies
uv sync --locked
```

**Slow responses**:
- Check network connectivity
- Verify Jina API status
- Review timeout settings

**Memory issues**:
- Reduce `max_content_length` in read_webpage
- Limit `num_pages` in search_and_read
- Use `include_content=False` for large batches

### Performance Monitoring
The server logs performance metrics:
```
INFO:main:Search completed: 10 results in 1.23s
INFO:main:Webpage read successfully: 1500 words in 2.45s
INFO:main:Batch search completed: 3/3 successful in 4.56s
```

## 📈 Optimization Tips

1. **Use batch operations** for multiple queries
2. **Enable connection pooling** (default enabled)
3. **Set appropriate timeouts** for your use case
4. **Filter duplicates** to reduce processing
5. **Limit content length** for faster processing
6. **Use parallel processing** for independent requests

## 🤝 Support

- **Jina AI Documentation**: https://docs.jina.ai
- **MCP Protocol**: https://modelcontextprotocol.org
- **Performance Issues**: Check logs and connection status

## 📄 License

MIT License - See LICENSE file for details

---

**Built with ❤️ using Jina AI for blazing-fast web search and content extraction.**
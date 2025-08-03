# MCP Context7 Assistant Extension - Demo Guide

This document provides a step-by-step demo of the MCP Context7 Assistant extension functionality.

## 🎯 Prerequisites

Before running the demo, ensure you have the following services running:

### 1. Nomic Embed Server
```bash
# Install nomic
pip install nomic[local]

# Start the embedding server
python -m nomic.embed.server --port 8080
```

### 2. Neo4j Database
```bash
# Using Docker (recommended)
docker run --name neo4j \
  -p7474:7474 -p7687:7687 \
  -d \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Check it's running
curl http://localhost:7474
```

## 🚀 Demo Steps

### Step 1: Install and Activate Extension

1. Open VSCode
2. Install the MCP Context7 Assistant extension
3. Reload VSCode to activate the extension
4. Check that the extension is active in the Extensions panel

### Step 2: Configure Settings

Open VSCode Settings (`Ctrl+,`) and configure:

```json
{
  "mcpContext7.neo4jUri": "bolt://localhost:7687",
  "mcpContext7.neo4jUser": "neo4j", 
  "mcpContext7.neo4jPassword": "password",
  "mcpContext7.nomicEmbedUrl": "http://localhost:8080",
  "mcpContext7.embeddingModel": "nomic-embed-text-v1.5",
  "mcpContext7.workerThreads": 4,
  "mcpContext7.batchSize": 32,
  "mcpContext7.enableSimdJson": true
}
```

### Step 3: Embed Markdown Files

1. Open Command Palette (`Ctrl+Shift+P`)
2. Run: `MCP: Embed VSCode Markdown Files`
3. Watch the progress as the extension:
   - Finds markdown files (especially copilot.md, claude.md)
   - Generates embeddings using Nomic Embed
   - Stores embeddings and metadata in Neo4j
4. View the results panel showing embedded files and statistics

**Expected Output:**
```
✅ Embedded 2 markdown files (15 chunks, 15 stored) in 2340ms
```

### Step 4: Search Evidence

1. Open Command Palette (`Ctrl+Shift+P`) 
2. Run: `MCP: Search Evidence`
3. Enter a search query, for example:
   - "agent orchestration"
   - "legal document analysis" 
   - "worker threads performance"
4. View semantic search results with similarity scores
5. Click through results to see relevant text chunks

**Expected Output:**
```
🔍 Found 8 evidence items matching "agent orchestration"
```

### Step 5: Parse Evidence JSON

1. Create a sample evidence JSON file:

```json
{
  "evidence": [
    {
      "id": "evidence_001",
      "type": "document", 
      "content": "This contract outlines the terms and conditions for legal AI services...",
      "metadata": {
        "source": "contract_analysis",
        "timestamp": "2024-01-15T10:30:00Z",
        "author": "Legal Team",
        "case_id": "CASE_2024_001"
      }
    },
    {
      "id": "evidence_002",
      "type": "testimony",
      "content": "The witness testified that they observed the defendant...",
      "metadata": {
        "source": "witness_statement", 
        "timestamp": "2024-01-16T14:15:00Z",
        "witness_name": "Jane Smith"
      }
    }
  ]
}
```

2. Open Command Palette (`Ctrl+Shift+P`)
3. Run: `MCP: Parse Evidence SIMD JSON`
4. Choose "📁 Select JSON file" and select your evidence file
5. View parsing results with performance metrics

**Expected Output:**
```
✅ Parsed 2 evidence items in 45ms using simd
```

## 📊 Demo Verification

### Check Neo4j Database

1. Open Neo4j Browser: http://localhost:7474
2. Login with neo4j/password
3. Run query to see stored embeddings:

```cypher
MATCH (e:Embedding) RETURN e.source, e.file_path, e.chunk_index LIMIT 10
```

### Performance Metrics

Monitor the extension performance:
- Embedding speed: ~100ms per document chunk
- Search speed: <50ms for similarity queries
- SIMD parsing: 3-5x faster than native JSON.parse

## 🎯 Expected Demo Flow

1. **Initialization** (30 seconds)
   - Extension activates
   - Services connect
   - Configuration validated

2. **Embedding** (1-2 minutes)
   - Markdown files discovered
   - Text chunked and embedded
   - Stored in Neo4j with metadata

3. **Search** (5-10 seconds)
   - Query processed
   - Embeddings generated
   - Similarity search executed
   - Results displayed

4. **Parsing** (5-15 seconds)  
   - JSON data loaded
   - SIMD parsing executed
   - Evidence extracted and validated
   - Results stored (optional)

## 🔧 Troubleshooting Demo Issues

### "Services not initialized" Error
- Check that Nomic Embed server is running on port 8080
- Verify Neo4j is accessible on port 7687
- Restart VSCode extension

### "No markdown files found"
- Ensure copilot.md and claude.md exist in workspace
- Check file permissions
- Try with other .md files

### "SIMD parsing failed"
- Verify JSON syntax is valid
- Check that simdjson module installed: `npm install simdjson`
- Try with smaller JSON files first

### Performance Issues
- Reduce worker thread count in settings
- Lower batch size for memory constraints
- Check system resources (CPU/Memory)

## 📈 Demo Success Metrics

A successful demo should achieve:

- ✅ All 3 commands execute without errors
- ✅ Markdown files embedded in <3 minutes
- ✅ Search returns relevant results in <10 seconds
- ✅ JSON parsing completes with SIMD optimization
- ✅ Neo4j database contains embedded data
- ✅ Performance metrics show expected speeds

## 🎉 Demo Conclusion

The MCP Context7 Assistant extension successfully demonstrates:

1. **Multi-core Embedding**: Parallel processing of markdown documentation
2. **Semantic Search**: AI-powered evidence discovery using embeddings
3. **High-Performance Parsing**: SIMD-optimized JSON processing for legal data
4. **Graph Database Integration**: Structured storage and retrieval with Neo4j
5. **VSCode Integration**: Seamless workflow within developer environment

This showcases a complete AI workflow for legal document processing that can be expanded for production use with additional features like:
- Automated document classification
- Advanced query interfaces
- Integration with external legal databases
- Multi-agent orchestration workflows
- Real-time collaboration features
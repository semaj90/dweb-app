import fetch from 'node-fetch';

async function testEnhancedRAGInsight() {
  try {
    // Test the enhanced-rag-insight tool via MCP server
    const response = await fetch('http://localhost:4000/mcp/call', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tool: 'enhanced-rag-insight',
        arguments: {
          query: 'contract liability analysis',
          context: 'legal AI system',
          documentType: 'contract'
        }
      })
    });

    if (response.ok) {
      const result = await response.text();
      console.log('✅ Enhanced RAG Insight Test - SUCCESS');
      console.log(result);
    } else {
      console.log('❌ Enhanced RAG Insight Test - HTTP ERROR:', response.status);
      console.log(await response.text());
    }
  } catch (error) {
    console.log('❌ Enhanced RAG Insight Test - ERROR:', error.message);
  }
}

testEnhancedRAGInsight();

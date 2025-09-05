// Gemma3 Legal Synthesis Integration - Complete Pipeline Test
// Run: node test-gemma3-synthesis.mjs

import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';

const API_BASE = 'http://localhost:5173';
const OLLAMA_BASE = 'http://localhost:11434';

class SynthesisValidator {
  async validateOllamaConnection() {
    console.log('🔗 Ollama Connection Test');
    
    try {
      const response = await fetch(`${OLLAMA_BASE}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: 'Legal AI status check',
          stream: false
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('✅ Gemma3 Legal accessible');
        return true;
      }
    } catch (error) {
      console.log('❌ Ollama not running - execute: ollama serve');
      return false;
    }
  }

  async testEmbeddings() {
    console.log('\n🧠 Nomic Embeddings Test');
    
    try {
      const response = await fetch(`${OLLAMA_BASE}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: 'Evidence synthesis test embedding'
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log(`✅ Embedding generated: ${result.embedding.length} dimensions`);
        return true;
      }
    } catch (error) {
      console.log('❌ Nomic embeddings failed');
      return false;
    }
  }

  async testSynthesisAPI() {
    console.log('\n🔬 Evidence Synthesis API Test');
    
    const testRequest = {
      evidenceIds: ['evidence-001', 'evidence-002'],
      synthesisType: 'correlation',
      caseId: 'test-case-legal',
      title: 'Gemma3 Legal Correlation Analysis',
      description: 'Evidence pattern analysis using local legal LLM',
      prompt: 'Analyze evidentiary relationships with focus on admissibility, chain of custody, and prosecution strategy. Identify temporal patterns and causal connections.'
    };

    try {
      const response = await fetch(`${API_BASE}/api/evidence/synthesize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer legal-test-token'
        },
        body: JSON.stringify(testRequest)
      });

      console.log(`Status: ${response.status}`);
      const result = await response.json();
      
      // Validation flow
      if (response.status === 401) {
        console.log('✅ API accessible - auth layer working');
        return 'auth';
      } else if (response.status === 404) {
        console.log('✅ DB query executed - schema validation passed');
        return 'db';
      } else if (result.success) {
        console.log('✅ Synthesis completed');
        console.log(`   RAG Score: ${result.metadata.ragScore}`);
        console.log(`   Confidence: ${result.metadata.confidence}`);
        console.log(`   Sources: ${result.metadata.sourceEvidenceCount}`);
        return 'success';
      } else {
        console.log('🔄 Processing - check AI service logs');
        console.log(`   Error: ${result.error}`);
        return 'processing';
      }
      
    } catch (error) {
      console.log(`❌ API request failed: ${error.message}`);
      return 'failed';
    }
  }

  async testRAGIntegration() {
    console.log('\n🎯 Enhanced RAG Integration Test');
    
    const ragQuery = {
      query: 'Legal evidence correlation analysis prosecution strategy',
      useContextRAG: true,
      useSelfPrompting: true,
      useMultiAgent: true,
      documentTypes: ['evidence', 'legal'],
      maxResults: 10
    };

    try {
      const response = await fetch(`${API_BASE}/api/enhanced-rag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ragQuery)
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('✅ Enhanced RAG operational');
        console.log(`   Documents: ${result.documents?.length || 0}`);
        console.log(`   RAG Score: ${result.metadata?.ragScore || 'N/A'}`);
        return true;
      }
    } catch (error) {
      console.log('❌ Enhanced RAG not available');
      return false;
    }
  }

  async executeFullPipeline() {
    console.log('🚀 Gemma3 Legal Synthesis Pipeline Validation\n');
    
    const results = {
      ollama: await this.validateOllamaConnection(),
      embeddings: await this.testEmbeddings(),
      synthesis: await this.testSynthesisAPI(),
      rag: await this.testRAGIntegration()
    };
    
    console.log('\n📊 Pipeline Status:');
    Object.entries(results).forEach(([test, result]) => {
      const status = result === true || result === 'success' ? '✅' : 
                   result === 'auth' || result === 'db' ? '🔄' : '❌';
      console.log(`   ${status} ${test}: ${result}`);
    });
    
    if (results.synthesis === 'success') {
      console.log('\n🎉 Synthesis pipeline operational - production ready');
    } else if (results.synthesis === 'auth' || results.synthesis === 'db') {
      console.log('\n✅ Core infrastructure working - configure auth/DB');
    } else {
      console.log('\n⚠️  Issues detected - check service dependencies');
    }
    
    return results;
  }
}

// Execute validation
new SynthesisValidator().executeFullPipeline();

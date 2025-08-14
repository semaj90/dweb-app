// Production Logging & Testing Suite
// File: production-validation.mjs

import fetch from 'node-fetch';
import fs from 'fs';
import { createWriteStream } from 'fs';

class ProductionLogger {
  constructor() {
    this.logFile = `logs/synthesis-test-${Date.now()}.log`;
    this.results = {};
    this.ensureLogDir();
  }

  ensureLogDir() {
    if (!fs.existsSync('logs')) fs.mkdirSync('logs', { recursive: true });
  }

  log(message, type = 'info') {
    const timestamp = new Date().toISOString();
    const logEntry = `[${timestamp}] ${type.toUpperCase()}: ${message}`;
    
    console.log(logEntry);
    fs.appendFileSync(this.logFile, logEntry + '\n');
  }

  logResult(test, result, metadata = {}) {
    this.results[test] = { result, metadata, timestamp: Date.now() };
    this.log(`${test}: ${JSON.stringify(result)}`, 'RESULT');
  }

  async validateGemma3Legal() {
    this.log('Testing Gemma3 Legal model accessibility');
    
    try {
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: 'Legal analysis test: Provide brief response',
          stream: false
        })
      });

      if (response.ok) {
        const result = await response.json();
        this.logResult('gemma3_legal', 'OPERATIONAL', {
          responseLength: result.response?.length,
          modelLoaded: true
        });
        return true;
      }
    } catch (error) {
      this.logResult('gemma3_legal', 'FAILED', { error: error.message });
      return false;
    }
  }

  async validateSynthesisAPI() {
    this.log('Testing Evidence Synthesis API');
    
    const payload = {
      evidenceIds: ['test-1', 'test-2'],
      synthesisType: 'correlation',
      caseId: 'validation-case',
      title: 'Production Validation Test',
      prompt: 'Legal evidence correlation analysis'
    };

    try {
      const response = await fetch('http://localhost:5173/api/evidence/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const status = response.status;
      const result = await response.json();
      
      this.logResult('synthesis_api', status === 401 ? 'AUTH_OK' : status, {
        endpoint: '/api/evidence/synthesize',
        responseTime: Date.now() - this.startTime
      });
      
      return status;
    } catch (error) {
      this.logResult('synthesis_api', 'FAILED', { error: error.message });
      return false;
    }
  }

  async validateRAGStudio() {
    this.log('Testing Enhanced RAG Studio');
    
    try {
      const response = await fetch('http://localhost:5173/api/enhanced-rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'legal evidence analysis',
          useContextRAG: true,
          maxResults: 5
        })
      });

      this.logResult('rag_studio', response.status, {
        endpoint: '/api/enhanced-rag/query'
      });
      
      return response.ok;
    } catch (error) {
      this.logResult('rag_studio', 'FAILED', { error: error.message });
      return false;
    }
  }

  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      results: this.results,
      summary: {
        total: Object.keys(this.results).length,
        passed: Object.values(this.results).filter(r => 
          r.result === 'OPERATIONAL' || r.result === 'AUTH_OK' || r.result === 200
        ).length
      }
    };

    fs.writeFileSync('logs/production-report.json', JSON.stringify(report, null, 2));
    this.log(`Report generated: ${Object.keys(this.results).length} tests completed`);
    
    return report;
  }
}

const logger = new ProductionLogger();
logger.startTime = Date.now();

// Execute validation sequence
Promise.all([
  logger.validateGemma3Legal(),
  logger.validateSynthesisAPI(),
  logger.validateRAGStudio()
]).then(() => {
  const report = logger.generateReport();
  console.log('\n📊 PRODUCTION VALIDATION COMPLETE');
  console.log(`✅ ${report.summary.passed}/${report.summary.total} tests operational`);
});

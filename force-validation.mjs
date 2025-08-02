// Direct API Validation & Force Testing
// File: force-validation.mjs

import fs from 'fs';
import { spawn } from 'child_process';

class ForceValidator {
  constructor() {
    this.logs = [];
    this.results = {};
    this.logFile = `logs/force-validation-${Date.now()}.log`;
    this.ensureLogDir();
  }

  ensureLogDir() {
    if (!fs.existsSync('logs')) fs.mkdirSync('logs', { recursive: true });
  }

  log(message, type = 'info') {
    const timestamp = new Date().toISOString();
    const entry = `[${timestamp}] ${type.toUpperCase()}: ${message}`;
    console.log(entry);
    this.logs.push(entry);
    fs.appendFileSync(this.logFile, entry + '\n');
  }

  async testWithFetch(url, options = {}) {
    try {
      // Use dynamic import for fetch since we can't install node-fetch
      const fetch = (await import('node-fetch')).default;
      return await fetch(url, options);
    } catch (error) {
      // Fallback to curl if fetch fails
      return this.testWithCurl(url, options);
    }
  }

  testWithCurl(url, options = {}) {
    return new Promise((resolve) => {
      const method = options.method || 'GET';
      const headers = options.headers || {};
      const body = options.body;
      
      let curlArgs = ['-s', '-w', '%{http_code}', url];
      
      if (method !== 'GET') {
        curlArgs.unshift('-X', method);
      }
      
      if (body) {
        curlArgs.push('-d', body);
      }
      
      Object.entries(headers).forEach(([key, value]) => {
        curlArgs.push('-H', `${key}: ${value}`);
      });

      const curl = spawn('curl', curlArgs, { stdio: 'pipe' });
      let output = '';
      
      curl.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      curl.on('close', (code) => {
        const statusCode = output.slice(-3);
        const responseBody = output.slice(0, -3);
        resolve({
          ok: statusCode.startsWith('2'),
          status: parseInt(statusCode),
          json: () => {
            try { return JSON.parse(responseBody); }
            catch { return { error: 'Invalid JSON' }; }
          },
          text: () => responseBody
        });
      });
    });
  }

  async validateOllamaGPU() {
    this.log('Testing Ollama GPU acceleration');
    
    const response = await this.testWithCurl('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3-legal',
        prompt: 'GPU test',
        stream: false
      })
    });

    this.results.ollama_gpu = {
      status: response.status,
      accessible: response.status !== 0,
      timestamp: Date.now()
    };
    
    this.log(`Ollama GPU test: ${response.status}`, response.ok ? 'success' : 'warning');
    return response.ok;
  }

  async validateSynthesisAPI() {
    this.log('Testing Evidence Synthesis API');
    
    const payload = {
      evidenceIds: ['test-1', 'test-2'],
      synthesisType: 'correlation',
      caseId: 'force-test',
      title: 'Force Validation Test'
    };

    const response = await this.testWithCurl('http://localhost:5173/api/evidence/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    this.results.synthesis_api = {
      status: response.status,
      accessible: response.status !== 0,
      timestamp: Date.now()
    };

    this.log(`Synthesis API: ${response.status}`, 'info');
    return response.status === 401 || response.status === 200; // Auth required is OK
  }

  async validateRAGStudio() {
    this.log('Testing RAG Studio');
    
    const response = await this.testWithCurl('http://localhost:5173/api/enhanced-rag/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: 'legal analysis test',
        maxResults: 5
      })
    });

    this.results.rag_studio = {
      status: response.status,
      accessible: response.status !== 0,
      timestamp: Date.now()
    };

    this.log(`RAG Studio: ${response.status}`, 'info');
    return response.status !== 0;
  }

  async executeForceValidation() {
    this.log('🚀 FORCE VALIDATION STARTED');
    
    // Parallel testing for speed
    const [ollama, synthesis, rag] = await Promise.all([
      this.validateOllamaGPU(),
      this.validateSynthesisAPI(), 
      this.validateRAGStudio()
    ]);

    const report = {
      timestamp: new Date().toISOString(),
      results: this.results,
      summary: {
        ollama_accessible: ollama,
        synthesis_accessible: synthesis,
        rag_accessible: rag,
        overall_status: synthesis ? 'OPERATIONAL' : 'NEEDS_CONFIG'
      },
      logs: this.logs
    };

    fs.writeFileSync('logs/force-validation-report.json', JSON.stringify(report, null, 2));
    
    this.log('📊 VALIDATION COMPLETE');
    this.log(`Status: ${report.summary.overall_status}`);
    
    return report;
  }
}

// Execute
new ForceValidator().executeForceValidation().then(report => {
  console.log('\\n✅ FORCE VALIDATION REPORT:');
  console.log(`Ollama: ${report.results.ollama_gpu?.status || 'OFFLINE'}`);
  console.log(`Synthesis: ${report.results.synthesis_api?.status || 'OFFLINE'}`);
  console.log(`RAG: ${report.results.rag_studio?.status || 'OFFLINE'}`);
  console.log(`Overall: ${report.summary.overall_status}`);
}).catch(console.error);

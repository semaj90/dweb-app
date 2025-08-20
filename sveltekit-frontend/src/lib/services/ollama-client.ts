// Unified client wrapper for server Ollama AI endpoints
// Frontend code should use this instead of directly accessing local Ollama.

interface GenerateOptions {
  system?: string;
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  model?: string;
}

export class OllamaClient {
  baseUrl = '/api/ai';

  async generate(prompt: string, options: GenerateOptions = {}) {
    const res = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: prompt, options })
    });
    if (!res.ok) throw new Error(`Generate failed: ${res.statusText}`);
    return res.json();
  }

  async legalAnalyze(documentId: string, analysisType = 'full') {
    const res = await fetch('/api/legal/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ documentId, analysisType })
    });
    if (!res.ok) throw new Error('Legal analysis failed');
    return res.json();
  }

  async systemStatus() {
    const res = await fetch('/api/system/status');
    if (!res.ok) throw new Error('Status failed');
    return res.json();
  }
}

export const ollamaClient = new OllamaClient();
export default ollamaClient;

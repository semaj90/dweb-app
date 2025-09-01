// src/lib/llm/gemma.ts
// Gemma3 helper for your local Ollama setup

export interface GemmaOptions {
  model?: string;
  maxTokens?: number;
  temperature?: number;
  stream?: boolean;
}

export interface GemmaResponse {
  text: string;
  tokens?: number;
  model?: string;
  done?: boolean;
}

export async function queryGemma(prompt: string, opts: GemmaOptions = {}): Promise<string> {
  const API = import.meta.env.GEMMA3_API_URL ?? 'http://localhost:11434';
  const model = opts.model ?? 'gemma3-legal';
  
  const body = {
    model,
    prompt,
    stream: false,
    options: {
      num_predict: opts.maxTokens ?? 512,
      temperature: opts.temperature ?? 0.0
    }
  };

  try {
    const res = await fetch(`${API}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`Gemma3 error: ${res.status} ${text}`);
    }

    const data = await res.json();
    
    // Handle different response formats from Ollama
    return data?.response ?? data?.text ?? data?.output ?? JSON.stringify(data);
  } catch (error: any) {
    console.error('Gemma3 query failed:', error);
    throw new Error(`Gemma3 connection failed: ${error.message}`);
  }
}

// Stream response version (for future use)
export async function* streamGemma(prompt: string, opts: GemmaOptions = {}): AsyncGenerator<string> {
  const API = import.meta.env.GEMMA3_API_URL ?? 'http://localhost:11434';
  const model = opts.model ?? 'gemma3-legal';
  
  const body = {
    model,
    prompt,
    stream: true,
    options: {
      num_predict: opts.maxTokens ?? 512,
      temperature: opts.temperature ?? 0.0
    }
  };

  const res = await fetch(`${API}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    throw new Error(`Gemma3 stream error: ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());

      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.response) {
            yield data.response;
          }
          if (data.done) return;
        } catch {
          // Skip invalid JSON lines
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
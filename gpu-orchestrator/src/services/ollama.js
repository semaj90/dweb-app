/**
 * Ollama Service - AI model integration
 */
export class OllamaService {
  constructor(endpoint) {
    this.endpoint = endpoint;
    this.models = new Map();
  }

  async initialize() {
    try {
      const response = await fetch(`${this.endpoint}/api/tags`);
      const data = await response.json();
      
      data.models.forEach(model => {
        this.models.set(model.name, model);
      });

      console.log(`✅ Ollama connected with ${data.models.length} models`);
      return true;
    } catch (error) {
      console.error('Ollama initialization failed:', error);
      throw error;
    }
  }

  async generateResponse(options) {
    const { model, prompt, stream = false, options: modelOptions = {} } = options;

    try {
      const response = await fetch(`${this.endpoint}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          prompt,
          stream,
          options: {
            num_gpu: 35, // Use all available GPU layers
            temperature: 0.1,
            top_p: 0.9,
            ...modelOptions
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        response: data.response,
        model: data.model,
        created_at: data.created_at,
        done: data.done,
        total_duration: data.total_duration,
        load_duration: data.load_duration,
        prompt_eval_count: data.prompt_eval_count,
        eval_count: data.eval_count
      };
    } catch (error) {
      console.error('Ollama generation error:', error);
      throw error;
    }
  }

  async generateEmbedding(text, model = 'nomic-embed-text') {
    try {
      const response = await fetch(`${this.endpoint}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          prompt: text
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama embeddings error: ${response.statusText}`);
      }

      const data = await response.json();
      return data.embedding;
    } catch (error) {
      console.error('Ollama embedding error:', error);
      throw error;
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(`${this.endpoint}/api/tags`);
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  getAvailableModels() {
    return Array.from(this.models.keys());
  }
}
import axios from "axios";


// This function calls your local Ollama server to get an embedding
export async function getEmbedding(text: string): Promise<number[]> {
  try {
    // Note: We call the Vite proxy URL, not Ollama directly
    const response = await axios.post('http://localhost:5173/api/llm/api/embeddings', {
      model: 'nomic-embed-text',
      prompt: text,
    });
    return response.data.embedding;
  } catch (error) {
    console.error("Failed to get embedding from Ollama:", error);
    throw new Error("Embedding generation failed.");
  }
}
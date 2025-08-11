import { apiFetch } from "../clients/api-client";

export async function embed(text: string) {
  return apiFetch("http://localhost:11434/api/embeddings", "POST", {
    body: { model: "nomic-embed-text", prompt: text, stream: false },
  });
}

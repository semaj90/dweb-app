import { apiFetch } from "../clients/api-client";

export async function generate(prompt: string) {
  return apiFetch("http://localhost:11434/api/generate", "POST", {
    body: { model: "gemma3-legal:latest", prompt, stream: false },
  });
}

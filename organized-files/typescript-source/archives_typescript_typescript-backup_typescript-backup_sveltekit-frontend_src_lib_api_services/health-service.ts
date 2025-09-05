import { apiFetch } from "../clients/api-client";

export async function getHealth(): Promise<{ status: string }> {
  return apiFetch("/health");
}

export async function getOllamaVersion(): Promise<{ version: string }> {
  return apiFetch("http://localhost:11434/api/version");
}

import { apiFetch } from "../clients/api-client";

export async function getHealth(): Promise<{ status: string }> {
  return apiFetch("http://localhost:8080/health");
}

export async function getOllamaVersion(): Promise<{ version: string }> {
  return apiFetch("http://localhost:11434/api/version");
}

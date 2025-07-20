import { invoke } from "@tauri-apps/api/tauri";

export async function getAvailableModels(): Promise<string[]> {
  return await invoke<string[]>("list_llm_models");
}
export async function runInference(
  model: string,
  prompt: string,
): Promise<string> {
  return await invoke<string>("run_llm_inference", { model, prompt });
}

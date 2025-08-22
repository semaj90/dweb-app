
import { writable } from "svelte/store";
// Orphaned content: import {

export class AIAssistant {
  private isLoading = writable(false);
  private response = writable("");

  async queryOllama(prompt: string) {
    this.isLoading.set(true);
    try {
      const response = await fetch("http://localhost:11434/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemma3:latest",
          prompt,
          stream: false,
        }),
      });
      const data = await response.json();
      this.response.set(data.response);
    } catch (error) {
      this.response.set("Error connecting to AI");
    } finally {
      this.isLoading.set(false);
    }
  }

  getStores() {
    return { isLoading: this.isLoading, response: this.response };
  }
}

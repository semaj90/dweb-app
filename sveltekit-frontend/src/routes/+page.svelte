<!-- Ask AI Component Refactored for Production Use -->
<script lang="ts">
  import { browser } from "$app/environment";
  import { createEventDispatcher, onMount } from "svelte";
  import { speakWithCoqui, loadCoquiTTS } from '$lib/services/coquiTTS';
  import type { Case } from '$lib/types';
  import { page } from '$app/stores';
  import { Button } from '$lib/components/ui';
  import { Brain, Search, Loader2, AlertCircle, CheckCircle, MessageCircle } from "lucide-svelte/icons";

  export let caseId: string | undefined;
  export let evidenceIds: string[] = [];
  export const placeholder = "Ask AI about this case...";
  export const maxHeight = "400px";
  export let showReferences = true;
  export const enableVoiceInput = false;
  export const enableVoiceOutput = false;

  interface AIResponse {
    answer: string;
    references: Array<{ id: string; type: string; title: string; relevanceScore: number; citation: string }>;
    confidence: number;
    searchResults: number;
    model: string;
    processingTime: number;
  }

  interface ConversationMessage {
    id: string;
    type: "user" | "ai";
    content: string;
    timestamp: number;
    references?: AIResponse["references"];
    confidence?: number;
    metadata?: Record<string, any>;
  }

  // Component State
  let query = "";
  let isLoading = false;
  let error = "";
  let conversation: ConversationMessage[] = [];
  let textareaRef: HTMLTextAreaElement;
  let messagesContainer: HTMLDivElement;

  // Advanced AI controls
  let showAdvancedOptions = false;
  let selectedModel: "openai" | "ollama" = "openai";
  let searchThreshold = 0.7;
  let maxResults = 10;
  let temperature = 0.7;

  let isListening = false;
  let recognition: any = null;
  let ttsLoading = false;
  let audioContext: AudioContext | null = null;

  const dispatch = createEventDispatcher();

  function generateId() {
    return crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
  }

  function scrollToBottom() {
    messagesContainer?.scrollTo({ top: messagesContainer.scrollHeight, behavior: 'smooth' });
  }

  async function askAI() {
    if (!query.trim() || isLoading) return;

    const userMessage: ConversationMessage = {
      id: generateId(),
      type: "user",
      content: query.trim(),
      timestamp: Date.now(),
    };
    conversation = [...conversation, userMessage];
    const aiMessageId = generateId();
    let aiMessage: ConversationMessage = {
      id: aiMessageId,
      type: "ai",
      content: "",
      timestamp: Date.now(),
      references: [],
      metadata: {},
    };
    conversation = [...conversation, aiMessage];
    query = "";
    isLoading = true;

    try {
      const requestBody = {
        question: userMessage.content,
        context: { caseId, evidenceIds, maxResults, searchThreshold },
        options: { model: selectedModel, temperature, maxTokens: 1000, includeReferences: showReferences },
      };

      const endpoint = selectedModel === "ollama" ? "/api/ai/chat" : "/api/ai/ask";
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) throw new Error("Failed to fetch AI response");

      if (selectedModel === "ollama" && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let done = false;

        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;
          buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const chunk = JSON.parse(line);
              aiMessage.content += chunk.answer || "";
              aiMessage.confidence = chunk.confidence;
              aiMessage.references = chunk.references || [];
              aiMessage.metadata = {
                model: chunk.model,
                processingTime: chunk.processingTime,
                searchResults: chunk.searchResults,
              };
              conversation = conversation.map((m) => m.id === aiMessageId ? { ...aiMessage } : m);
              scrollToBottom();
            } catch {}
          }
        }
      } else {
        const result = await response.json();
        aiMessage = {
          ...aiMessage,
          content: result.answer,
          references: result.references,
          confidence: result.confidence,
          metadata: {
            model: result.model,
            processingTime: result.processingTime,
            searchResults: result.searchResults,
          }
        };
        conversation = conversation.map((m) => m.id === aiMessageId ? aiMessage : m);
      }
    } catch (err) {
      error = (err as Error).message;
    } finally {
      isLoading = false;
    }
  }
</script>

<!-- Markup & Style are handled in the UI layer component at src/lib/components/ui/AskAI.svelte -->

<script lang="ts">
  interface Props {
    height?: any;
    caseId: string | undefined ;
  }
  let {
    height = "500px",
    caseId = undefined
  } = $props();



  import { Button } from "$lib/components/ui/button";
  import { Textarea } from "$lib/components/ui/textarea/index";
  import {
    aiPersonality,
    chatActions,
    currentConversation,
    isLoading,
    isTyping,
    showProactivePrompt,
  } from "$lib/stores/chatStore";
  import { notifications } from "$lib/stores/notification";
  import type { ApiResponse, ChatRequest, ChatResponse } from "$lib/types/api";
  import { Bot, Loader2, Send } from "lucide-svelte";
  import { onDestroy, onMount, tick } from "svelte";
  import ChatMessage from "./ChatMessage.svelte";
  import ProactivePrompt from "./ProactivePrompt.svelte";
  import ThinkingStyleToggle from "./ThinkingStyleToggle.svelte";
  import { ThinkingProcessor } from "$lib/ai/thinking-processor";

    
  let messageInput = "";
  let messagesContainer: HTMLElement
  let inputElement: HTMLTextAreaElement
  let inactivityTimer: NodeJS.Timeout;

  // Enhanced thinking style state
  let thinkingStyleEnabled = false;
  let analysisMode = false;
  let lastAnalysisResult: any = null;

  const IDLE_TIMEOUT = 60000; // 60 seconds

  function handleUserActivity() {
    clearTimeout(inactivityTimer);
    chatActions.updateActivity();

    inactivityTimer = setTimeout(() => {
      triggerProactivePrompt();
    }, IDLE_TIMEOUT);
}
  function triggerProactivePrompt() {
    if ($currentConversation && $currentConversation.messages.length > 0) {
      showProactivePrompt.set(true);
}
}
  async function sendMessage() {
    if (!messageInput.trim()) return;

    const userMessage = messageInput.trim();
    messageInput = "";

    // Reset activity timer
    handleUserActivity();

    // Add user message
    chatActions.addMessage(userMessage, "user");

    try {
      chatActions.setLoading(true);
      chatActions.setTyping(true);

      // Check if this is an analysis request
      const isAnalysisRequest = userMessage.toLowerCase().includes('analyze') ||
                               userMessage.toLowerCase().includes('evidence') ||
                               userMessage.toLowerCase().includes('case');

      let response: Response

      if (isAnalysisRequest && (caseId || thinkingStyleEnabled)) {
        // Use the enhanced analysis endpoint
        response = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: userMessage,
            caseId,
            useThinkingStyle: thinkingStyleEnabled,
            analysisType: 'reasoning',
            documentType: 'legal_document'
          }),
        });
      } else {
        // Use the regular chat endpoint
        const requestBody: ChatRequest = {
          messages: $currentConversation?.messages || [],
          context: {
            caseId,
            currentPage: window.location.pathname,
            thinkingStyle: thinkingStyleEnabled
          },
        };

        response = await fetch("/api/ai/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        });
}
      if (!response.ok) {
        throw new Error("Failed to get AI response");
}
      const apiResponse = await response.json();

      if (!apiResponse.success) {
        throw new Error(apiResponse.error || "Invalid response format");
}
      // Handle different response types
      if (apiResponse.analysis) {
        // This is an analysis response
        lastAnalysisResult = apiResponse.analysis;
        analysisMode = true;

        const content = formatAnalysisResponse(apiResponse.analysis, apiResponse.metadata);
        chatActions.addMessage(content, "assistant", {
          ...apiResponse.metadata,
          analysisResult: apiResponse.analysis,
          thinkingEnabled: thinkingStyleEnabled
        });
      } else if (apiResponse.data) {
        // This is a regular chat response
        chatActions.addMessage(
          apiResponse.data.content,
          "assistant",
          apiResponse.data.metadata
        );
}
      // Scroll to bottom
      setTimeout(scrollToBottom, 100);
    } catch (error) {
      console.error("Chat error:", error);
      notifications.add({
        type: "error",
        title: "Chat Error",
        message: "Failed to get response from AI assistant",
      });
    } finally {
      chatActions.setLoading(false);
      chatActions.setTyping(false);
}
}
  function formatAnalysisResponse(analysis: any, metadata: any): string {
    if (!analysis) return "Analysis completed.";

    let response = `# AI Analysis Results\n\n`;

    // Add thinking process if available
    if (analysis.thinking && thinkingStyleEnabled) {
      response += `## 🧠 Reasoning Process\n\n`;
      response += `*Showing step-by-step AI reasoning:*\n\n`;
      response += analysis.thinking.replace(/\n/g, '\n\n') + `\n\n`;
      response += `---\n\n`;
}
    // Add main analysis
    response += `## 📋 Analysis Results\n\n`;

    if (analysis.analysis) {
      const analysisData = analysis.analysis;

      if (analysisData.key_findings) {
        response += `**Key Findings:**\n`;
        analysisData.key_findings.forEach((finding: string) => {
          response += `• ${finding}\n`;
        });
        response += `\n`;
}
      if (analysisData.legal_implications) {
        response += `**Legal Implications:**\n`;
        analysisData.legal_implications.forEach((implication: string) => {
          response += `• ${implication}\n`;
        });
        response += `\n`;
}
      if (analysisData.recommendations) {
        response += `**Recommendations:**\n`;
        analysisData.recommendations.forEach((rec: string) => {
          response += `• ${rec}\n`;
        });
        response += `\n`;
}
      if (analysisData.raw_analysis) {
        response += analysisData.raw_analysis + `\n\n`;
}
}
    // Add confidence and metadata
    response += `## 📊 Analysis Metadata\n\n`;
    response += `• **Confidence:** ${Math.round(analysis.confidence * 100)}%\n`;
    response += `• **Model:** ${metadata.model_used}\n`;
    response += `• **Processing Time:** ${metadata.processing_time}ms\n`;
    response += `• **Thinking Style:** ${metadata.thinking_enabled ? 'Enabled' : 'Disabled'}\n`;

    if (analysis.reasoning_steps && analysis.reasoning_steps.length > 0) {
      response += `\n**Reasoning Steps:**\n`;
      analysis.reasoning_steps.forEach((step: string, index: number) => {
        response += `${index + 1}. ${step}\n`;
      });
}
    return response;
}
  async function handleProactiveResponse() {
    if (!$showProactivePrompt || !$currentConversation) return;

    try {
      chatActions.setLoading(true);
      chatActions.setTyping(true);
      showProactivePrompt.set(false);

      const requestBody: ChatRequest = {
        messages: $currentConversation.messages,
        context: {
          caseId,
          currentPage: window.location.pathname,
          thinkingStyle: thinkingStyleEnabled
        },
        proactiveMode: true,
      };

      const response = await fetch("/api/ai/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) throw new Error("Failed to get proactive response");

      const apiResponse: ApiResponse<ChatResponse> = await response.json();

      if (!apiResponse.success || !apiResponse.data) {
        throw new Error(apiResponse.error || "Invalid response format");
}
      chatActions.addMessage(apiResponse.data.content, "assistant", {
        ...apiResponse.data.metadata,
        proactive: true,
      });

      setTimeout(scrollToBottom, 100);
    } catch (error) {
      console.error("Proactive response error:", error);
    } finally {
      chatActions.setLoading(false);
      chatActions.setTyping(false);
}
}
  function handleThinkingToggle(event: CustomEvent<{ enabled: boolean }>) {
    thinkingStyleEnabled = event.detail.enabled;

    // Add a system message to indicate the change
    const message = thinkingStyleEnabled
      ? "🧠 Thinking Style enabled. AI will now show detailed reasoning process."
      : "⚡ Quick Mode enabled. AI will provide concise responses.";

    notifications.add({
      type: "info",
      title: "AI Mode Changed",
      message,
    });
}
  async function quickAnalyzeEvidence() {
    if (!caseId) {
      notifications.add({
        type: "warning",
        title: "No Case Selected",
        message: "Please select a case to analyze evidence.",
      });
      return;
}
    try {
      const analysis = await ThinkingProcessor.analyzeCase(caseId, {
        analysisType: 'reasoning',
        useThinkingStyle: thinkingStyleEnabled
      });

      const content = formatAnalysisResponse(analysis, analysis.metadata);
      chatActions.addMessage(content, "assistant", {
        ...analysis.metadata,
        analysisResult: analysis,
        quickAction: true
      });

      setTimeout(scrollToBottom, 100);
    } catch (error) {
      console.error("Quick analysis error:", error);
      notifications.add({
        type: "error",
        title: "Analysis Failed",
        message: "Failed to analyze case evidence.",
      });
}
}
  function scrollToBottom() {
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
}
  function handleKeyDown(event: CustomEvent<KeyboardEvent>) {
    const keyEvent = event.detail;
    if (keyEvent.key === "Enter" && !keyEvent.shiftKey) {
      event.preventDefault();
      sendMessage();
}
    handleUserActivity();
}
  function autoResize() {
    if (inputElement) {
      inputElement.style.height = "auto";
      inputElement.style.height =
        Math.min(inputElement.scrollHeight, 120) + "px";
}
}
  onMount(() => {
    // Initialize conversation if none exists
    if (!$currentConversation) {
      chatActions.newConversation(caseId ? `Case ${caseId}` : undefined);
}
    // Set up activity tracking
    handleUserActivity();

    // Add global activity listeners
    window.addEventListener("mousemove", handleUserActivity);
    window.addEventListener("keydown", handleUserActivity);
    window.addEventListener("click", handleUserActivity);

    // Auto-focus input
    if (inputElement) {
      inputElement.focus();
}
  });

  onDestroy(() => {
    clearTimeout(inactivityTimer);
    window.removeEventListener("mousemove", handleUserActivity);
    window.removeEventListener("keydown", handleUserActivity);
    window.removeEventListener("click", handleUserActivity);
  });

  // Reactive scroll to bottom when new messages arrive
  $effect(() => { if ($currentConversation?.messages) {
    tick().then(scrollToBottom);
}
</script>

<div class="space-y-4">
  <!-- Enhanced Header with Thinking Toggle -->
  <div class="space-y-4">
    <div class="space-y-4">
      <div class="space-y-4">
        <ThinkingStyleToggle
          bind:enabled={thinkingStyleEnabled}
          loading={$isLoading}
          premium={true}
          size="sm"
          ontoggle={handleThinkingToggle}
        />

        {#if caseId}
          <Button
            variant="outline"
            size="sm"
            onclick={quickAnalyzeEvidence}
            disabled={$isLoading}
          >
            🔍 Quick Analysis
          </Button>
        {/if}
      </div>

      <div class="space-y-4">
        {#if lastAnalysisResult}
          <span class="space-y-4">
            📊 Confidence: {Math.round(lastAnalysisResult.confidence * 100)}%
          </span>
        {/if}
        <div class="space-y-4">
          <div class="space-y-4"></div>
          <span>AI Active</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Messages Container -->
  <div
    bind:this={messagesContainer}
    class="space-y-4"
    style="height: calc({height} - 140px);"
  >
    {#if $currentConversation?.messages.length === 0}
      <!-- Enhanced Welcome Message -->
      <div class="space-y-4">
        <div
          class="space-y-4"
        >
          <Bot class="space-y-4" />
        </div>
        <h3 class="space-y-4">
          Hi! I'm {$aiPersonality.name}, your enhanced AI legal assistant
        </h3>
        <p class="space-y-4">
          I can provide both quick responses and detailed reasoning analysis.
          Toggle thinking style above to see my step-by-step reasoning process.
        </p>

        {#if thinkingStyleEnabled}
          <div class="space-y-4">
            <p class="space-y-4">
              🧠 <strong>Thinking Style Active:</strong> I'll show my reasoning process for deeper analysis.
            </p>
          </div>
        {:else}
          <div class="space-y-4">
            <p class="space-y-4">
              ⚡ <strong>Quick Mode Active:</strong> I'll provide fast, concise responses.
            </p>
          </div>
        {/if}
      </div>
    {:else}
      <!-- Messages -->
      {#each ($currentConversation?.messages || []) as message (message.id)}
        <ChatMessage {message} />
      {/each}
    {/if}

    <!-- Typing Indicator -->
    {#if $isTyping}
      <div class="space-y-4">
        <div
          class="space-y-4"
        >
          <Bot class="space-y-4" />
        </div>
        <div class="space-y-4">
          <div class="space-y-4">
            <div class="space-y-4"></div>
            <div
              class="space-y-4"
              style="animation-delay: 0.1s"
            ></div>
            <div
              class="space-y-4"
              style="animation-delay: 0.2s"
            ></div>
          </div>
          {#if thinkingStyleEnabled}
            <p class="space-y-4">Thinking step by step...</p>
          {/if}
        </div>
      </div>
    {/if}
  </div>

  <!-- Proactive Prompt -->
  {#if $showProactivePrompt}
    <div class="space-y-4">
      <ProactivePrompt
        onaccept={handleProactiveResponse}
        ondismiss={() => showProactivePrompt.set(false)}
      />
    </div>
  {/if}

  <!-- Input Area -->
  <div class="space-y-4">
    <div class="space-y-4">
      <div class="space-y-4">
        <Textarea
          bind:element={inputElement}
          bind:value={messageInput}
          placeholder={thinkingStyleEnabled
            ? "Ask for detailed analysis... (Enter to send, Shift+Enter for new line)"
            : "Type your message... (Enter to send, Shift+Enter for new line)"}
          class="space-y-4"
          onkeydown={handleKeyDown}
          oninput={autoResize}
          disabled={$isLoading}
        />
      </div>

      <Button
        variant="default"
        size="sm"
        class="space-y-4"
        onclick={() => sendMessage()}
        disabled={$isLoading || !messageInput.trim()}
      >
        {#if $isLoading}
          <Loader2 class="space-y-4" />
        {:else}
          <Send class="space-y-4" />
        {/if}
      </Button>
    </div>

    <!-- Enhanced Status Text -->
    <div class="space-y-4">
      <div class="space-y-4">
        {#if ($currentConversation?.messages?.length || 0) > 0}
          <span>{$currentConversation?.messages?.length || 0} messages</span>
        {/if}
        {#if caseId}
          <span>• Case: {caseId}</span>
        {/if}
        {#if analysisMode}
          <span>• Analysis Mode</span>
        {/if}
      </div>

      <div class="space-y-4">
        <span class="space-y-4">
          {thinkingStyleEnabled ? "🧠 Thinking" : "⚡ Quick"}
        </span>
      </div>
    </div>
  </div>
</div>

<style>
  /* @unocss-include */
  :global(.message-content p) {
    margin-bottom: 0.5rem;
}
  :global(.message-content p:last-child) {
    margin-bottom: 0;
}
  :global(.message-content ul, .message-content ol) {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
}
  :global(.message-content code) {
    background: rgba(0, 0, 0, 0.1);
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    font-family: "Courier New", monospace;
}
  :global(.message-content h1, .message-content h2, .message-content h3) {
    font-weight: 600;
    margin: 1rem 0 0.5rem 0;
}
  :global(.message-content h1) {
    font-size: 1.25rem;
}
  :global(.message-content h2) {
    font-size: 1.125rem;
}
  :global(.message-content h3) {
    font-size: 1rem;
}
</style>

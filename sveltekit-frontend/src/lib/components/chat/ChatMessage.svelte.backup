<script lang="ts">
  import type { ChatMessage } from "$lib/stores/chatStore";
  import DOMPurify from "dompurify";
  import { Bot } from "lucide-svelte";
  import "./chat-message.css";

  export let message: ChatMessage;
  const sanitizedContent = DOMPurify.sanitize(message.content);
</script>

<div class="container mx-auto px-4" class:user={message.role === "user"}>
  <div class="container mx-auto px-4">
    {#if message.role === "assistant"}
      <Bot class="container mx-auto px-4" />
    {:else}
      <span class="container mx-auto px-4">You</span>
    {/if}
  </div>
  <div class="container mx-auto px-4">
    {@html sanitizedContent}
  </div>
</div>

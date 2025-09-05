<script lang="ts">
  interface Props {
    message: ChatMessage
  }
  let {
    message
  } = $props();



  import type { ChatMessage } from "$lib/stores/chatStore";
  import DOMPurify from "dompurify";
  import { Bot } from "lucide-svelte";
  import "./chat-message.css";

    const sanitizedContent = DOMPurify.sanitize(message.content);
</script>

<div class="space-y-4" class:user={message.role === "user"}>
  <div class="space-y-4">
    {#if message.role === "assistant"}
      <Bot class="space-y-4" />
    {:else}
      <span class="space-y-4">You</span>
    {/if}
  </div>
  <div class="space-y-4">
    {@html sanitizedContent}
  </div>
</div>

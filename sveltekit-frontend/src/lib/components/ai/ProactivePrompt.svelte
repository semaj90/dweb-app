<script lang="ts">

  interface Props {
    onaccept?: (event?: unknown) => void;
    ondismiss?: (event?: unknown) => void;
    onquickResponse?: (event?: unknown) => void;
  }


  import { Button } from "$lib/components/ui/button";
  import { aiPersonality } from "$lib/stores/chatStore";
  import { Clock, Lightbulb, MessageCircle, Sparkles, X } from "lucide-svelte";
  
  
  // Array of proactive prompts based on context
  const proactivePrompts = [
    "Would you like me to help clarify anything we've discussed?",
    "I notice we've been working on this for a while. Need a different approach?",
    "Is there anything specific you'd like me to focus on?",
    "Would you like me to summarize what we've covered so far?",
    "Any questions about the legal concepts we've discussed?",
    "Should we explore this topic from a different angle?",
    "Would you like some additional resources on this subject?",
    "Is there a particular aspect you'd like to dive deeper into?",
  ];

  // Get a random proactive prompt
  const randomPrompt =
    proactivePrompts[Math.floor(Math.random() * proactivePrompts.length)];

  function handleAccept() {
    onaccept?.();
}
  function handleDismiss() {
    ondismiss?.();
}
</script>

<div
  class="space-y-4"
>
  <!-- Header -->
  <div class="space-y-4">
    <!-- AI Avatar with pulse animation -->
    <div class="space-y-4">
      <div class="space-y-4">
        <div
          class="space-y-4"
        >
          <Sparkles class="space-y-4" />
        </div>
        <!-- Pulse ring -->
        <div
          class="space-y-4"
        ></div>
      </div>
    </div>

    <!-- Content -->
    <div class="space-y-4">
      <!-- Header -->
      <div class="space-y-4">
        <Clock class="space-y-4" />
        <span class="space-y-4">
          {$aiPersonality.name} here!
        </span>
      </div>

      <!-- Message -->
      <p class="space-y-4">
        {randomPrompt}
      </p>

      <!-- Actions -->
      <div class="space-y-4">
        <!-- Accept Button -->
        <Button
          variant="outline"
          size="sm"
          class="space-y-4"
          on:on:click={() => handleAccept()}
        >
          <MessageCircle class="space-y-4" />
          Yes, help me
        </Button>

        <!-- Quick responses -->
        <Button
          variant="ghost"
          size="sm"
          class="space-y-4"
          on:on:click={() => onquickResponse?.()}
        >
          <Lightbulb class="space-y-4" />
          Summarize
        </Button>

        <!-- Dismiss Button -->
        <Button
          variant="ghost"
          size="sm"
          class="space-y-4"
          on:on:click={() => handleDismiss()}
          title="Not now"
        >
          <X class="space-y-4" />
        </Button>
      </div>
    </div>
  </div>

  <!-- Subtle progress indicator -->
  <div class="space-y-4">
    <div
      class="space-y-4"
    ></div>
  </div>
</div>

<style>
  /* @unocss-include */
  @keyframes slide-in-from-bottom {
    from {
      transform: translateY(100%);
      opacity: 0;
}
    to {
      transform: translateY(0);
      opacity: 1;
}}
  .animate-in {
    animation-fill-mode: both;
}
  .slide-in-from-bottom {
    animation-name: slide-in-from-bottom;
}
  .duration-300 {
    animation-duration: 300ms;
}
</style>

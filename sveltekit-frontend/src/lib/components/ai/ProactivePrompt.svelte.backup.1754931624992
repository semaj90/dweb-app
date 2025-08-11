<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import { aiPersonality } from "$lib/stores/chatStore";
  import { Clock, Lightbulb, MessageCircle, Sparkles, X } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";

  const dispatch = createEventDispatcher();

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
    dispatch("accept");
}
  function handleDismiss() {
    dispatch("dismiss");
}
</script>

<div
  class="container mx-auto px-4"
>
  <!-- Header -->
  <div class="container mx-auto px-4">
    <!-- AI Avatar with pulse animation -->
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <div
          class="container mx-auto px-4"
        >
          <Sparkles class="container mx-auto px-4" />
        </div>
        <!-- Pulse ring -->
        <div
          class="container mx-auto px-4"
        ></div>
      </div>
    </div>

    <!-- Content -->
    <div class="container mx-auto px-4">
      <!-- Header -->
      <div class="container mx-auto px-4">
        <Clock class="container mx-auto px-4" />
        <span class="container mx-auto px-4">
          {$aiPersonality.name} here!
        </span>
      </div>

      <!-- Message -->
      <p class="container mx-auto px-4">
        {randomPrompt}
      </p>

      <!-- Actions -->
      <div class="container mx-auto px-4">
        <!-- Accept Button -->
        <Button
          variant="outline"
          size="sm"
          class="container mx-auto px-4"
          on:click={() => handleAccept()}
        >
          <MessageCircle class="container mx-auto px-4" />
          Yes, help me
        </Button>

        <!-- Quick responses -->
        <Button
          variant="ghost"
          size="sm"
          class="container mx-auto px-4"
          on:click={() => dispatch("quickResponse", "summarize")}
        >
          <Lightbulb class="container mx-auto px-4" />
          Summarize
        </Button>

        <!-- Dismiss Button -->
        <Button
          variant="ghost"
          size="sm"
          class="container mx-auto px-4"
          on:click={() => handleDismiss()}
          title="Not now"
        >
          <X class="container mx-auto px-4" />
        </Button>
      </div>
    </div>
  </div>

  <!-- Subtle progress indicator -->
  <div class="container mx-auto px-4">
    <div
      class="container mx-auto px-4"
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

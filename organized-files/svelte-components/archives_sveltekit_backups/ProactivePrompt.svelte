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
  class="mx-auto px-4 max-w-7xl"
>
  <!-- Header -->
  <div class="mx-auto px-4 max-w-7xl">
    <!-- AI Avatar with pulse animation -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <div
          class="mx-auto px-4 max-w-7xl"
        >
          <Sparkles class="mx-auto px-4 max-w-7xl" />
        </div>
        <!-- Pulse ring -->
        <div
          class="mx-auto px-4 max-w-7xl"
        ></div>
      </div>
    </div>

    <!-- Content -->
    <div class="mx-auto px-4 max-w-7xl">
      <!-- Header -->
      <div class="mx-auto px-4 max-w-7xl">
        <Clock class="mx-auto px-4 max-w-7xl" />
        <span class="mx-auto px-4 max-w-7xl">
          {$aiPersonality.name} here!
        </span>
      </div>

      <!-- Message -->
      <p class="mx-auto px-4 max-w-7xl">
        {randomPrompt}
      </p>

      <!-- Actions -->
      <div class="mx-auto px-4 max-w-7xl">
        <!-- Accept Button -->
        <Button
          variant="outline"
          size="sm"
          class="mx-auto px-4 max-w-7xl"
          onclick={() => handleAccept()}
        >
          <MessageCircle class="mx-auto px-4 max-w-7xl" />
          Yes, help me
        </Button>

        <!-- Quick responses -->
        <Button
          variant="ghost"
          size="sm"
          class="mx-auto px-4 max-w-7xl"
          onclick={() => dispatch("quickResponse", "summarize")}
        >
          <Lightbulb class="mx-auto px-4 max-w-7xl" />
          Summarize
        </Button>

        <!-- Dismiss Button -->
        <Button
          variant="ghost"
          size="sm"
          class="mx-auto px-4 max-w-7xl"
          onclick={() => handleDismiss()}
          title="Not now"
        >
          <X class="mx-auto px-4 max-w-7xl" />
        </Button>
      </div>
    </div>
  </div>

  <!-- Subtle progress indicator -->
  <div class="mx-auto px-4 max-w-7xl">
    <div
      class="mx-auto px-4 max-w-7xl"
    ></div>
  </div>
</div>

<style>
  @keyframes slide-in-from-bottom {
    from {
      transform: translateY(100%);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

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

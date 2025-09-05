<script lang="ts">
  interface Props {
    currentStep: number
    totalSteps: number
    stepLabels: string[] ;
  }
  let {
    currentStep,
    totalSteps,
    stepLabels = []
  } = $props();



      let { stepLabels = $bindable() } = $props(); // string[] = [];

  function getStepStatus(stepIndex: number): 'completed' | 'current' | 'upcoming' {
    if (stepIndex < currentStep) return 'completed';
    if (stepIndex === currentStep) return 'current';
    return 'upcoming';
  }

  function getStepColor(status: string): string {
    switch (status) {
      case 'completed': return 'bg-green-500 border-green-500 text-white';
      case 'current': return 'bg-blue-500 border-blue-500 text-white';
      case 'upcoming': return 'bg-gray-200 border-gray-300 text-gray-600';
      default: return 'bg-gray-200 border-gray-300 text-gray-600';
    }
  }

  function getConnectorColor(stepIndex: number): string {
    return stepIndex < currentStep ? 'bg-green-500' : 'bg-gray-300';
  }

  const defaultLabels = [
    'Case Info',
    'Documents',
    'Evidence',
    'AI Analysis',
    'Review'
  ];

  let labels = $derived(stepLabels.length > 0 ? stepLabels : defaultLabels)
</script>

<div class="py-6">
  <nav aria-label="Progress">
    <ol role="list" class="flex items-center">
      {#each labels as label, index}
        {@const status = getStepStatus(index)}

        <li class="relative {index !== labels.length - 1 ? 'pr-8 sm:pr-20' : ''}">
          <!-- Step Circle -->
          <div class="flex items-center">
            <div class="relative flex h-8 w-8 items-center justify-center rounded-full border-2 {getStepColor(status)} transition-all duration-200">
              {#if status === 'completed'}
                <svg class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
              {:else}
                <span class="text-sm font-medium">{index + 1}</span>
              {/if}
            </div>

            <!-- Step Label -->
            <div class="ml-3">
              <p class="text-sm font-medium {status === 'current' ? 'text-blue-600' : status === 'completed' ? 'text-green-600' : 'text-gray-500'}">
                {label}
              </p>
            </div>
          </div>

          <!-- Connector Line -->
          {#if index !== labels.length - 1}
            <div class="absolute top-4 left-4 -ml-px mt-0.5 h-full w-0.5 {getConnectorColor(index + 1)} transition-all duration-200" aria-hidden="true"></div>
          {/if}
        </li>
      {/each}
    </ol>
  </nav>

  <!-- Progress Bar -->
  <div class="mt-6">
    <div class="bg-gray-200 rounded-full h-2">
      <div
        class="bg-blue-500 h-2 rounded-full transition-all duration-500 ease-out"
        style="width: {((currentStep) / (totalSteps - 1)) * 100}%"
      ></div>
    </div>
    <div class="flex justify-between text-xs text-gray-500 mt-1">
      <span>Start</span>
      <span>{Math.round(((currentStep) / (totalSteps - 1)) * 100)}% Complete</span>
      <span>Complete</span>
    </div>
  </div>
</div>

<style>
  /* Ensure smooth transitions and proper spacing */
  li {
    min-width: 0;
  }

  @media (max-width: 640px) {
    /* Responsive adjustments for mobile */
    li {
      padding-right: 1rem;
    }

    .ml-3 {
      margin-left: 0.5rem;
    }
  }
</style>

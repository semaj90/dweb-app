<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Dropdown from '$lib/components/+Dropdown.svelte';
  import Checkbox from '$lib/components/+Checkbox.svelte';

  const dispatch = createEventDispatcher();

  let selectedAutomationType: string = '';
  let selectedSource: string = '';
  let enableAutoProcessing: boolean = false;

  const handleSubmit = async () => {
    // In a real application, you would send this data to your backend API
    // For now, we'll just log it and dispatch an event.
    console.log('Automate Upload Data:', {
      selectedAutomationType,
      selectedSource,
      enableAutoProcessing,
    });

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));

    alert('Automation settings saved!');
    dispatch('automationSuccess');

    // Reset form
    selectedAutomationType = '';
    selectedSource = '';
    enableAutoProcessing = false;
  };


  // Fetch automation type options from API (simulated async fetch)
  import { onMount } from 'svelte';
  let automationTypeOptions: { value: string; label: string }[] = [];
  let loadingAutomationTypes = true;

  async function fetchAutomationTypes() {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));
    automationTypeOptions = [
      { value: 'folder_watch', label: 'Folder Watch' },
      { value: 'email_attachment', label: 'Email Attachment' },
      { value: 'api_integration', label: 'API Integration' },
    ];
    loadingAutomationTypes = false;
  }

  onMount(() => {
    fetchAutomationTypes();
  });

  const sourceOptions = [
    { value: 'source1', label: 'Shared Drive A' },
    { value: 'source2', label: 'Outlook Inbox' },
    { value: 'source3', label: 'External API' },
  ];
</script>


<div class="rounded-xl bg-nier-surface shadow-lg p-8 max-w-lg mx-auto">
  <div class="border-b border-nier-border pb-4 mb-6">
    <h3 class="text-xl font-bold nier-text-glow">Automate Upload</h3>
  </div>
  <div class="space-y-6">
    <div>
      <label for="automationTypeSelect" class="block font-semibold mb-2 text-nier-white">Automation Type:</label>
      <Dropdown id="automationTypeSelect" bind:selected={selectedAutomationType} options={automationTypeOptions} class="w-full" />
    </div>
    <div>
      <label for="sourceSelect" class="block font-semibold mb-2 text-nier-white">Source:</label>
      <Dropdown id="sourceSelect" bind:selected={selectedSource} options={sourceOptions} class="w-full" />
    </div>
    <div class="flex items-center gap-2">
      <Checkbox id="autoProcessCheckbox" bind:checked={enableAutoProcessing} label="Enable Auto-Processing" />
    </div>
    <button class="btn btn-primary w-full mt-4" on:click={handleSubmit}>
      Save Automation
    </button>
  </div>
</div>


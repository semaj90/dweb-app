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

  // Dummy data for dropdowns - replace with actual data fetched from API
  const automationTypeOptions = [
    { value: 'folder_watch', label: 'Folder Watch' },
    { value: 'email_attachment', label: 'Email Attachment' },
    { value: 'api_integration', label: 'API Integration' },
  ];

  const sourceOptions = [
    { value: 'source1', label: 'Shared Drive A' },
    { value: 'source2', label: 'Outlook Inbox' },
    { value: 'source3', label: 'External API' },
  ];
</script>

<div class="card">
  <div class="card-header">
    <h3>Automate Upload</h3>
  </div>
  <div class="card-body">
    <div class="mb-3">
      <label for="automationTypeSelect" class="form-label">Automation Type:</label>
      <Dropdown id="automationTypeSelect" bind:selected={selectedAutomationType} options={automationTypeOptions} />
    </div>
    <div class="mb-3">
      <label for="sourceSelect" class="form-label">Source:</label>
      <Dropdown id="sourceSelect" bind:selected={selectedSource} options={sourceOptions} />
    </div>
    <div class="mb-3 form-check">
      <Checkbox id="autoProcessCheckbox" bind:checked={enableAutoProcessing} label="Enable Auto-Processing" />
    </div>
    <button class="btn btn-primary" on:click={handleSubmit}>Save Automation</button>
  </div>
</div>

<style>
  .card {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 1.5rem;
  }

  .card-header {
    border-bottom: 1px solid #eee;
    padding-bottom: 1rem;
    margin-bottom: 1rem;
  }

  .card-header h3 {
    margin: 0;
    font-size: 1.25rem;
    color: #333;
  }

  .form-label {
    font-weight: bold;
    margin-bottom: 0.5rem;
    display: block;
  }

  .form-control {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 1rem;
  }

  .btn-primary {
    background-color: #007bff;
    color: #fff;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
  }

  .btn-primary:hover {
    background-color: #0056b3;
  }
</style>
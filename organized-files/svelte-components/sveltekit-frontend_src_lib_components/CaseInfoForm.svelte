<script lang="ts">
  interface Props {
    formData: {
  }
  let {
    formData
  } = $props();



  import { createEventDispatcher } from 'svelte';
  import { Button } from 'bits-ui';
  import { fade } from 'svelte/transition';

  const dispatch = createEventDispatcher();

      title: string
    client_name: string
    case_type: string
    jurisdiction: string
    priority: 'low' | 'medium' | 'high' | 'urgent';
    description: string
    key_dates: Array<{ date: string description: string }>;
  };

  let validationErrors: Record<string, string> = {};

  // Case type options
  const caseTypes = [
    'Civil Litigation',
    'Criminal Defense',
    'Corporate Law',
    'Family Law',
    'Real Estate',
    'Intellectual Property',
    'Employment Law',
    'Personal Injury',
    'Contract Dispute',
    'Administrative Law'
  ];

  // Jurisdiction options
  const jurisdictions = [
    'Federal Court',
    'State Court - California',
    'State Court - New York',
    'State Court - Texas',
    'State Court - Florida',
    'Arbitration',
    'Mediation',
    'Administrative Agency',
    'International'
  ];

  function validateForm() {
    validationErrors = {};

    if (!formData.title.trim()) {
      validationErrors.title = 'Case title is required';
    }

    if (!formData.client_name.trim()) {
      validationErrors.client_name = 'Client name is required';
    }

    if (!formData.case_type) {
      validationErrors.case_type = 'Case type is required';
    }

    if (!formData.jurisdiction) {
      validationErrors.jurisdiction = 'Jurisdiction is required';
    }

    if (!formData.description.trim()) {
      validationErrors.description = 'Case description is required';
    }

    return Object.keys(validationErrors).length === 0;
  }

  function addKeyDate() {
    formData.key_dates = [...formData.key_dates, { date: '', description: '' }];
  }

  function removeKeyDate(index: number) {
    formData.key_dates = formData.key_dates.filter((_, i) => i !== index);
  }

  function handleNext() {
    if (validateForm()) {
      dispatch('next', { step: 'caseInfo', data: formData });
    }
  }

  function handleSaveDraft() {
    dispatch('saveDraft', { step: 'caseInfo', data: formData });
  }

  // Priority colors
  function getPriorityColor(priority: string) {
    switch (priority) {
      case 'low': return 'bg-green-100 text-green-800 border-green-300';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'urgent': return 'bg-red-100 text-red-800 border-red-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  }
</script>

<div class="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg" transition:fade>
  <div class="mb-8">
    <h2 class="text-2xl font-bold text-gray-900 mb-2">Case Information</h2>
    <p class="text-gray-600">Enter the basic information about this legal case</p>
  </div>

  <form on:submit|preventDefault={handleNext} class="space-y-6">
    <!-- Case Title -->
    <div>
      <label for="title" class="block text-sm font-medium text-gray-700 mb-2">
        Case Title *
      </label>
      <input
        id="title"
        type="text"
        bind:value={formData.title}
        class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        class:border-red-500={validationErrors.title}
        placeholder="e.g., Smith vs. Jones Contract Dispute"
      />
      {#if validationErrors.title}
        <p class="mt-1 text-sm text-red-600">{validationErrors.title}</p>
      {/if}
    </div>

    <!-- Client Name -->
    <div>
      <label for="client_name" class="block text-sm font-medium text-gray-700 mb-2">
        Client Name *
      </label>
      <input
        id="client_name"
        type="text"
        bind:value={formData.client_name}
        class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        class:border-red-500={validationErrors.client_name}
        placeholder="Enter client's full name"
      />
      {#if validationErrors.client_name}
        <p class="mt-1 text-sm text-red-600">{validationErrors.client_name}</p>
      {/if}
    </div>

    <!-- Case Type and Priority Row -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label for="case_type" class="block text-sm font-medium text-gray-700 mb-2">
          Case Type *
        </label>
        <select
          id="case_type"
          bind:value={formData.case_type}
          class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          class:border-red-500={validationErrors.case_type}
        >
          <option value="">Select case type</option>
          {#each caseTypes as type}
            <option value={type}>{type}</option>
          {/each}
        </select>
        {#if validationErrors.case_type}
          <p class="mt-1 text-sm text-red-600">{validationErrors.case_type}</p>
        {/if}
      </div>

      <div>
        <label for="priority" class="block text-sm font-medium text-gray-700 mb-2">
          Priority Level
        </label>
        <select
          id="priority"
          bind:value={formData.priority}
          class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="low">Low Priority</option>
          <option value="medium">Medium Priority</option>
          <option value="high">High Priority</option>
          <option value="urgent">Urgent</option>
        </select>
        <div class="mt-2">
          <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border {getPriorityColor(formData.priority)}">
            {formData.priority.charAt(0).toUpperCase() + formData.priority.slice(1)} Priority
          </span>
        </div>
      </div>
    </div>

    <!-- Jurisdiction -->
    <div>
      <label for="jurisdiction" class="block text-sm font-medium text-gray-700 mb-2">
        Jurisdiction *
      </label>
      <select
        id="jurisdiction"
        bind:value={formData.jurisdiction}
        class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        class:border-red-500={validationErrors.jurisdiction}
      >
        <option value="">Select jurisdiction</option>
        {#each jurisdictions as jurisdiction}
          <option value={jurisdiction}>{jurisdiction}</option>
        {/each}
      </select>
      {#if validationErrors.jurisdiction}
        <p class="mt-1 text-sm text-red-600">{validationErrors.jurisdiction}</p>
      {/if}
    </div>

    <!-- Case Description -->
    <div>
      <label for="description" class="block text-sm font-medium text-gray-700 mb-2">
        Case Description *
      </label>
      <textarea
        id="description"
        bind:value={formData.description}
        rows="4"
        class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        class:border-red-500={validationErrors.description}
        placeholder="Provide a detailed description of the case, including key issues, parties involved, and relevant background information..."
      ></textarea>
      {#if validationErrors.description}
        <p class="mt-1 text-sm text-red-600">{validationErrors.description}</p>
      {/if}
    </div>

    <!-- Key Dates -->
    <div>
      <div class="flex items-center justify-between mb-3">
        <label class="block text-sm font-medium text-gray-700">
          Key Dates
        </label>
        <Button.Root
          type="button"
          onclick={addKeyDate}
          class="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          + Add Date
        </Button.Root>
      </div>

      {#each formData.key_dates as keyDate, index}
        <div class="flex gap-3 mb-3" transition:fade>
          <input
            type="date"
            bind:value={keyDate.date}
            class="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <input
            type="text"
            bind:value={keyDate.description}
            placeholder="Event description"
            class="flex-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <Button.Root
            type="button"
            onclick={() => removeKeyDate(index)}
            class="px-3 py-2 text-red-600 hover:text-red-800 focus:outline-none"
          >
            Remove
          </Button.Root>
        </div>
      {/each}

      {#if formData.key_dates.length === 0}
        <p class="text-sm text-gray-500 italic">No key dates added yet. Click "Add Date" to include important deadlines or milestones.</p>
      {/if}
    </div>

    <!-- Form Actions -->
    <div class="flex justify-between pt-6 border-t border-gray-200">
      <Button.Root
        type="button"
        onclick={handleSaveDraft}
        class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        Save Draft
      </Button.Root>

      <Button.Root
        type="submit"
        class="px-6 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        Next: Upload Documents →
      </Button.Root>
    </div>
  </form>
</div>

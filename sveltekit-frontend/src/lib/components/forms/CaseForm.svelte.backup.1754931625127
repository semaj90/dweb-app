<script lang="ts">
  import { caseFormSchema, type CaseForm } from "$lib/schemas/forms";
  import { getAuthContext } from "$lib/stores/auth";
  import { superForm, type SuperValidated } from 'sveltekit-superforms';
  import { zodClient } from 'sveltekit-superforms/adapters';

  interface Props {
    initialData?: SuperValidated<CaseForm> | Partial<CaseForm>;
    isEditing?: boolean;
    formApi?: any;
    onsuccess?: (data: any) => void;
    onerror?: (error: any) => void;
  }

  let {
    initialData = {},
    isEditing = false,
    formApi = $bindable(),
    onsuccess,
    onerror
  }: Props = $props();

  const auth = getAuthContext();

  // Available users for assignment (would come from API)
  let availableUsers = [
    { id: '1', name: 'John Smith', role: 'prosecutor' },
    { id: '2', name: 'Jane Doe', role: 'investigator' },
    { id: '3', name: 'Mike Johnson', role: 'legal_assistant' }
  ];

  // Initialize superForm
  const { form, errors, constraints, enhance, submitting, delayed, message } = superForm(initialData, {
    validators: zodClient(caseFormSchema),
    resetForm: false,
    invalidateAll: false,
    onSubmit: ({ cancel }) => {
      // You can add custom validation here
      console.log('Form submitted with data:', $form);
    },
    onResult: ({ result }) => {
      if (result.type === 'success') {
        onsuccess?.(result.data);
      } else if (result.type === 'error') {
        onerror?.(result.error);
      }
    }
  });

  // Update formApi when form changes using $effect
  $effect(() => {
    if (formApi !== undefined) {
      formApi = {
        form,
        errors,
        constraints,
        submitting,
        delayed,
        message,
        enhance
      };
    }
  });

  // Tag management
  let tagInput = '';

  function addTag() {
    if (tagInput.trim() && (!$form.tags || !$form.tags.includes(tagInput.trim()))) {
      $form.tags = [...($form.tags || []), tagInput.trim()];
      tagInput = '';
}}
  function removeTag(tag: string) {
    $form.tags = $form.tags?.filter(t => t !== tag) || [];
}
  function handleTagKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      event.preventDefault();
      addTag();
}}
  // Auto-generate case number
  function generateCaseNumber() {
    const year = new Date().getFullYear();
    const random = Math.floor(Math.random() * 900000) + 100000;
    $form.caseNumber = `CAS-${year}-${random}`;
}
</script>

<div class="container mx-auto px-4">
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div>
        <h2 class="container mx-auto px-4">
          {isEditing ? 'Edit Case' : 'Create New Case'}
        </h2>
        <p class="container mx-auto px-4">
          {isEditing ? 'Update case information and settings' : 'Enter case details to begin investigation'}
        </p>
      </div>

      {#if !isEditing}
        <button
          type="button"
          on:click={() => generateCaseNumber()}
          class="container mx-auto px-4"
        >
          Generate Case #
        </button>
      {/if}
    </div>

    <form method="POST" use:enhance>
      <!-- Case Number and Title -->
      <div class="container mx-auto px-4">
        <div>
          <label for="caseNumber" class="container mx-auto px-4">
            Case Number *
          </label>
          <input
            type="text"
            id="caseNumber"
            name="caseNumber"
            bind:value={$form.caseNumber}
            class="container mx-auto px-4"
            placeholder="e.g., CAS-2024-123456"
            {...$constraints.caseNumber}
          />
          {#if $errors.caseNumber}
            <p class="container mx-auto px-4">{$errors.caseNumber}</p>
          {/if}
        </div>

        <div>
          <label for="priority" class="container mx-auto px-4">
            Priority *
          </label>
          <select
            id="priority"
            name="priority"
            bind:value={$form.priority}
            class="container mx-auto px-4"
            {...$constraints.priority}
          >
            <option value="">Select priority</option>
            <option value="low">Low Priority</option>
            <option value="medium">Medium Priority</option>
            <option value="high">High Priority</option>
          </select>
          {#if $errors.priority}
            <p class="container mx-auto px-4">{$errors.priority}</p>
          {/if}
        </div>
      </div>

      <!-- Title -->
      <div class="container mx-auto px-4">
        <label for="title" class="container mx-auto px-4">
          Case Title *
        </label>
        <input
          type="text"
          id="title"
          name="title"
          bind:value={$form.title}
          class="container mx-auto px-4"
          placeholder="e.g., State vs. Johnson - Financial Fraud Investigation"
          {...$constraints.title}
        />
        {#if $errors.title}
          <p class="container mx-auto px-4">{$errors.title}</p>
        {/if}
      </div>

      <!-- Description -->
      <div class="container mx-auto px-4">
        <label for="description" class="container mx-auto px-4">
          Case Description
        </label>
        <textarea
          id="description"
          name="description"
          rows="4"
          bind:value={$form.description}
          class="container mx-auto px-4"
          placeholder="Provide a detailed description of the case..."
          {...$constraints.description}
        ></textarea>
        {#if $errors.description}
          <p class="container mx-auto px-4">{$errors.description}</p>
        {/if}
      </div>

      <!-- Status and Assignment -->
      <div class="container mx-auto px-4">
        <div>
          <label for="status" class="container mx-auto px-4">
            Status
          </label>
          <select
            id="status"
            name="status"
            bind:value={$form.status}
            class="container mx-auto px-4"
            {...$constraints.status}
          >
            <option value="draft">Draft</option>
            <option value="active">Active Investigation</option>
            <option value="pending">Pending Review</option>
            <option value="closed">Closed</option>
          </select>
        </div>

        <div>
          <label for="assignedTo" class="container mx-auto px-4">
            Assigned To
          </label>
          <select
            id="assignedTo"
            name="assignedTo"
            bind:value={$form.assignedTo}
            class="container mx-auto px-4"
            {...$constraints.assignedTo}
          >
            <option value="">Unassigned</option>
            {#each availableUsers as user}
              <option value={user.id}>{user.name} ({user.role})</option>
            {/each}
          </select>
        </div>
      </div>

      <!-- Due Date -->
      <div class="container mx-auto px-4">
        <label for="dueDate" class="container mx-auto px-4">
          Due Date
        </label>
        <input
          type="datetime-local"
          id="dueDate"
          name="dueDate"
          bind:value={$form.dueDate}
          class="container mx-auto px-4"
          {...$constraints.dueDate}
        />
        {#if $errors.dueDate}
          <p class="container mx-auto px-4">{$errors.dueDate}</p>
        {/if}
      </div>

      <!-- Tags -->
      <div class="container mx-auto px-4">
        <label for="tagInput" class="container mx-auto px-4">
          Tags
        </label>
        <div class="container mx-auto px-4">
          {#each $form.tags || [] as tag}
            <span class="container mx-auto px-4">
              {tag}
              <button
                type="button"
                on:click={() => removeTag(tag)}
                class="container mx-auto px-4"
              >
                ×
              </button>
            </span>
          {/each}
        </div>
        <div class="container mx-auto px-4">
          <input
            type="text"
            id="tagInput"
            bind:value={tagInput}
            on:keydown={handleTagKeydown}
            placeholder="Add a tag..."
            class="container mx-auto px-4"
            aria-label="Add tag"
          />
          <button
            type="button"
            on:click={() => addTag()}
            class="container mx-auto px-4"
          >
            Add Tag
          </button>
        </div>
        {#if $errors.tags}
          <p class="container mx-auto px-4">{$errors.tags}</p>
        {/if}
      </div>

      <!-- Checkboxes -->
      <div class="container mx-auto px-4">
        <label class="container mx-auto px-4">
          <input
            type="checkbox"
            bind:checked={$form.isConfidential}
            class="container mx-auto px-4"
          />
          <span class="container mx-auto px-4">Mark as confidential</span>
        </label>

        <label class="container mx-auto px-4">
          <input
            type="checkbox"
            bind:checked={$form.notifyAssignee}
            class="container mx-auto px-4"
          />
          <span class="container mx-auto px-4">Notify assignee via email</span>
        </label>
      </div>

      <!-- Submit Buttons -->
      <div class="container mx-auto px-4">
        <button
          type="button"
          on:click={() => dispatch('cancel')}
          class="container mx-auto px-4"
          disabled={$submitting}
        >
          Cancel
        </button>

        <button
          type="submit"
          class="container mx-auto px-4"
          disabled={$submitting}
        >
          {#if $submitting}
            <div class="container mx-auto px-4">
              <div class="container mx-auto px-4"></div>
              {isEditing ? 'Updating...' : 'Creating...'}
            </div>
          {:else}
            {isEditing ? 'Update Case' : 'Create Case'}
          {/if}
        </button>
      </div>

      <!-- Server Messages -->
      {#if $message}
        <div class="container mx-auto px-4">
          {$message.text}
        </div>
      {/if}

      <!-- Loading Indicator -->
      {#if $delayed}
        <div class="container mx-auto px-4">
          Processing your request...
        </div>
      {/if}
    </form>
  </div>
</div>

<style>
  /* @unocss-include */
  /* Custom validation styles */
  .legal-input:invalid {
    border-color: #ef4444;
}
  .legal-input:valid {
    border-color: #10b981;
}
</style>

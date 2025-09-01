<!--
  Enhanced Case Form with SuperForms + Zod + Actions Enhancement
  Demonstrates complete form enhancement pattern with validation
-->
<script lang="ts">
  import { enhance } from '$app/forms';
  import { superForm } from 'sveltekit-superforms/client';
  import { zod } from 'sveltekit-superforms/adapters';
  import { z } from 'zod';
  import { writable } from 'svelte/store';
  import Button from '$lib/components/ui/Button.svelte';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { Textarea } from '$lib/components/ui/textarea';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import * as Card from '$lib/components/ui/card';
  import * as Select from '$lib/components/ui/select';
  import { 
    AlertCircle, 
    Loader2, 
    Save, 
    CheckCircle, 
    Upload,
    FileText,
    Calendar,
    Users,
    Scale
  } from 'lucide-svelte';
  import { caseFormSchema, type CaseForm } from '$lib/schemas/forms';
  import { createCaseCreationForm } from '$lib/forms/superforms-xstate-integration';
  import type { SuperValidated } from 'sveltekit-superforms';

  // Svelte 5 Props Interface
  interface Props {
    data?: any; // SuperValidated<CaseForm>
    submitAction?: string;
    editMode?: boolean;
    enableAutoSave?: boolean;
    enableRealTimeValidation?: boolean;
    onsubmit?: (event: { data: CaseForm }) => void;
    onsuccess?: (event: { case: any }) => void;
    onerror?: (event: { message: string }) => void;
    ondraft?: (event: { data: CaseForm }) => void;
  }

  // Svelte 5 props with defaults
  let {
    data = undefined,
    submitAction = '?/createCase',
    editMode = false,
    enableAutoSave = true,
    enableRealTimeValidation = true,
    onsubmit,
    onsuccess,
    onerror,
    ondraft
  }: Props = $props();

  // Enhanced form integration with XState
  const formIntegration = createCaseCreationForm(data, {
    autoSave: enableAutoSave,
    autoSaveDelay: 2000,
    resetOnSuccess: !editMode,
    onSubmit: async (formData) => {
      if (onsubmit) onsubmit({ data: formData as CaseForm });
    },
    onSuccess: (result) => {
      if (onsuccess) onsuccess({ case: result });
    },
    onError: (error) => {
      if (onerror) onerror({ message: error });
      componentError = new Error(error);
    }
  });

  const { form, errors, enhance: formEnhance, submitting, message, delayed } = formIntegration.form;
  const { isValid, isSubmitting, progress } = formIntegration;

  // Local state using Svelte 5 runes
  let showAdvanced = $state(false);
  let uploadedFiles = $state<File[]>([]);
  let validationStatus = $state<'idle' | 'validating' | 'valid' | 'invalid'>('idle');
  let componentError = $state<Error | null>(null);

  // Priority levels with colors
  const priorityLevels = [
    { value: 'low', label: 'Low Priority', color: 'text-green-600' },
    { value: 'medium', label: 'Medium Priority', color: 'text-yellow-600' },
    { value: 'high', label: 'High Priority', color: 'text-red-600' }
  ];

  // Status options
  const statusOptions = [
    { value: 'draft', label: 'Draft', description: 'Case is being prepared' },
    { value: 'active', label: 'Active', description: 'Case is under investigation' },
    { value: 'pending', label: 'Pending', description: 'Awaiting review or action' },
    { value: 'closed', label: 'Closed', description: 'Case is completed' }
  ];

  // Enhanced form validation with real-time feedback
  $: {
    if (enableRealTimeValidation && $form) {
      validationStatus = 'validating';
      
      const validationResult = caseFormSchema.safeParse($form);
      
      setTimeout(() => {
        validationStatus = validationResult.success ? 'valid' : 'invalid';
      }, 300);
    }
  }

  // Auto-save indicator
  let lastSaved = $state<Date | null>(null);
  let isAutoSaving = $state(false);

  // Enhanced file upload handler
  function handleFileUpload(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.files) {
      uploadedFiles = [...uploadedFiles, ...Array.from(target.files)];
    }
  }

  // Remove uploaded file
  function removeFile(index: number) {
    uploadedFiles = uploadedFiles.filter((_, i) => i !== index);
  }

  // Format file size
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Enhanced form submission with progress tracking
  function createEnhancedSubmit() {
    return enhance(({ formData, action, cancel }) => {
      // Add uploaded files to form data
      uploadedFiles.forEach((file, index) => {
        formData.append(`attachments[${index}]`, file);
      });

      // Add metadata
      formData.append('metadata', JSON.stringify({
        submitTime: new Date().toISOString(),
        userAgent: navigator.userAgent,
        validationStatus,
        autoSaved: lastSaved !== null
      }));

      return async ({ result, update }) => {
        if (result.type === 'success') {
          // Handle success
          if (onsuccess) onsuccess({ case: result.data });
          
          // Reset form if not in edit mode
          if (!editMode) {
            uploadedFiles = [];
            lastSaved = null;
          }
        } else if (result.type === 'error') {
          // Handle error
          const errorMsg = result.error?.message || 'Submission failed';
          if (onerror) onerror({ message: errorMsg });
          componentError = new Error(errorMsg);
        }

        // Update the form
        await update();
      };
    });
  }
</script>

{#if !componentError}
<Card.Root class="w-full max-w-4xl mx-auto">
  <Card.Header>
    <div class="flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <Scale class="h-6 w-6 text-primary" />
        <div>
          <Card.Title class="text-xl">
            {editMode ? 'Edit Case' : 'Create New Case'}
          </Card.Title>
          <Card.Description>
            {editMode ? 'Update case information and evidence' : 'Enter case details and upload evidence'}
          </Card.Description>
        </div>
      </div>
      
      <!-- Progress indicator -->
      {#if $progress > 0}
        <div class="flex items-center space-x-2">
          <div class="w-20 bg-gray-200 rounded-full h-2">
            <div 
              class="bg-primary h-2 rounded-full transition-all duration-300"
              style="width: {$progress}%"
            ></div>
          </div>
          <span class="text-sm text-muted-foreground">{Math.round($progress)}%</span>
        </div>
      {/if}
    </div>
  </Card.Header>

  <Card.Content>
    <!-- Auto-save status -->
    {#if enableAutoSave && (lastSaved || isAutoSaving)}
      <div class="mb-4 p-3 bg-muted rounded-md flex items-center justify-between">
        <div class="flex items-center space-x-2">
          {#if isAutoSaving}
            <Loader2 class="h-4 w-4 animate-spin" />
            <span class="text-sm">Auto-saving...</span>
          {:else if lastSaved}
            <Save class="h-4 w-4 text-green-600" />
            <span class="text-sm">Last saved: {lastSaved.toLocaleTimeString()}</span>
          {/if}
        </div>
        
        <!-- Real-time validation status -->
        {#if enableRealTimeValidation}
          <div class="flex items-center space-x-2">
            {#if validationStatus === 'validating'}
              <Loader2 class="h-4 w-4 animate-spin text-yellow-600" />
              <span class="text-sm text-yellow-600">Validating...</span>
            {:else if validationStatus === 'valid'}
              <CheckCircle class="h-4 w-4 text-green-600" />
              <span class="text-sm text-green-600">Valid</span>
            {:else if validationStatus === 'invalid'}
              <AlertCircle class="h-4 w-4 text-red-600" />
              <span class="text-sm text-red-600">Issues found</span>
            {/if}
          </div>
        {/if}
      </div>
    {/if}

    <form 
      method="POST" 
      action={submitAction}
      use:createEnhancedSubmit()
      enctype="multipart/form-data"
      class="space-y-6"
    >
      <!-- Basic Information -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Case Number -->
        <div class="space-y-2">
          <Label for="caseNumber" class="flex items-center space-x-2">
            <FileText class="h-4 w-4" />
            <span>Case Number *</span>
          </Label>
          <Input
            id="caseNumber"
            name="caseNumber"
            placeholder="ABC-2024-123456"
            bind:value={$form.caseNumber}
            aria-invalid={$errors.caseNumber ? 'true' : undefined}
            class={$errors.caseNumber ? 'border-destructive' : ''}
          />
          {#if $errors.caseNumber}
            <p class="text-sm text-destructive flex items-center space-x-1">
              <AlertCircle class="h-3 w-3" />
              <span>{$errors.caseNumber[0]}</span>
            </p>
          {/if}
        </div>

        <!-- Priority -->
        <div class="space-y-2">
          <Label for="priority" class="flex items-center space-x-2">
            <AlertCircle class="h-4 w-4" />
            <span>Priority Level *</span>
          </Label>
          <Select.Root bind:selected={$form.priority} name="priority">
            <Select.Trigger class={$errors.priority ? 'border-destructive' : ''}>
              <Select.Value placeholder="Select priority" />
            </Select.Trigger>
            <Select.Content>
              {#each priorityLevels as priority}
                <Select.Item value={priority.value} class={priority.color}>
                  {priority.label}
                </Select.Item>
              {/each}
            </Select.Content>
          </Select.Root>
          {#if $errors.priority}
            <p class="text-sm text-destructive">{$errors.priority[0]}</p>
          {/if}
        </div>
      </div>

      <!-- Title -->
      <div class="space-y-2">
        <Label for="title">Case Title *</Label>
        <Input
          id="title"
          name="title"
          placeholder="Enter a descriptive case title"
          bind:value={$form.title}
          aria-invalid={$errors.title ? 'true' : undefined}
          class={$errors.title ? 'border-destructive' : ''}
        />
        {#if $errors.title}
          <p class="text-sm text-destructive">{$errors.title[0]}</p>
        {/if}
      </div>

      <!-- Description -->
      <div class="space-y-2">
        <Label for="description">Description</Label>
        <Textarea
          id="description"
          name="description"
          placeholder="Provide detailed case description (optional)"
          bind:value={$form.description}
          rows="4"
          aria-invalid={$errors.description ? 'true' : undefined}
          class={$errors.description ? 'border-destructive' : ''}
        />
        {#if $errors.description}
          <p class="text-sm text-destructive">{$errors.description[0]}</p>
        {/if}
        <p class="text-sm text-muted-foreground">
          {$form.description?.length || 0}/1000 characters
        </p>
      </div>

      <!-- Advanced Options -->
      <div class="border-t pt-6">
        <Button
          type="button"
          variant="ghost"
          on:on:on:click={() => showAdvanced = !showAdvanced}
          class="mb-4"
        >
          {showAdvanced ? 'Hide' : 'Show'} Advanced Options
        </Button>

        {#if showAdvanced}
          <div class="space-y-6 border-l-2 border-muted pl-6">
            <!-- Status and Assignment -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <!-- Status -->
              <div class="space-y-2">
                <Label for="status">Case Status</Label>
                <Select.Root bind:selected={$form.status} name="status">
                  <Select.Trigger>
                    <Select.Value placeholder="Select status" />
                  </Select.Trigger>
                  <Select.Content>
                    {#each statusOptions as status}
                      <Select.Item value={status.value}>
                        <div>
                          <div class="font-medium">{status.label}</div>
                          <div class="text-sm text-muted-foreground">{status.description}</div>
                        </div>
                      </Select.Item>
                    {/each}
                  </Select.Content>
                </Select.Root>
              </div>

              <!-- Due Date -->
              <div class="space-y-2">
                <Label for="dueDate" class="flex items-center space-x-2">
                  <Calendar class="h-4 w-4" />
                  <span>Due Date</span>
                </Label>
                <Input
                  id="dueDate"
                  name="dueDate"
                  type="datetime-local"
                  bind:value={$form.dueDate}
                  aria-invalid={$errors.dueDate ? 'true' : undefined}
                  class={$errors.dueDate ? 'border-destructive' : ''}
                />
                {#if $errors.dueDate}
                  <p class="text-sm text-destructive">{$errors.dueDate[0]}</p>
                {/if}
              </div>
            </div>

            <!-- Tags -->
            <div class="space-y-2">
              <Label for="tags">Tags (max 10)</Label>
              <Input
                id="tags"
                name="tags"
                placeholder="Enter tags separated by commas"
                bind:value={$form.tags}
              />
              <p class="text-sm text-muted-foreground">
                Use tags to categorize and organize cases
              </p>
            </div>

            <!-- Options -->
            <div class="flex flex-col space-y-3">
              <div class="flex items-center space-x-2">
                <Checkbox
                  id="isConfidential"
                  name="isConfidential"
                  bind:checked={$form.isConfidential}
                />
                <Label for="isConfidential">Mark as confidential</Label>
              </div>

              <div class="flex items-center space-x-2">
                <Checkbox
                  id="notifyAssignee"
                  name="notifyAssignee"
                  bind:checked={$form.notifyAssignee}
                />
                <Label for="notifyAssignee">Notify assignee when case is updated</Label>
              </div>
            </div>
          </div>
        {/if}
      </div>

      <!-- File Upload Section -->
      <div class="border-t pt-6">
        <div class="space-y-4">
          <div class="flex items-center space-x-2">
            <Upload class="h-5 w-5" />
            <Label class="text-base font-medium">Case Documents</Label>
          </div>

          <!-- File Upload Input -->
          <div class="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6">
            <div class="text-center">
              <Upload class="mx-auto h-12 w-12 text-muted-foreground/50" />
              <div class="mt-4">
                <Label for="file-upload" class="cursor-pointer">
                  <span class="text-sm font-medium text-primary">Upload files</span>
                  <input
                    id="file-upload"
                    type="file"
                    multiple
                    accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png"
                    onchange={handleFileUpload}
                    class="sr-only"
                  />
                </Label>
                <p class="text-sm text-muted-foreground">or drag and drop</p>
              </div>
              <p class="text-xs text-muted-foreground mt-2">
                PDF, DOC, DOCX, TXT, JPG, PNG up to 10MB each
              </p>
            </div>
          </div>

          <!-- Uploaded Files List -->
          {#if uploadedFiles.length > 0}
            <div class="space-y-2">
              <h4 class="text-sm font-medium">Uploaded Files ({uploadedFiles.length})</h4>
              <div class="space-y-2">
                {#each uploadedFiles as file, index}
                  <div class="flex items-center justify-between p-3 bg-muted rounded-md">
                    <div class="flex items-center space-x-3">
                      <FileText class="h-4 w-4 text-muted-foreground" />
                      <div>
                        <p class="text-sm font-medium">{file.name}</p>
                        <p class="text-xs text-muted-foreground">{formatFileSize(file.size)}</p>
                      </div>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      on:on:on:click={() => removeFile(index)}
                    >
                      Remove
                    </Button>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      </div>

      <!-- Form Actions -->
      <div class="flex items-center justify-between pt-6 border-t">
        <div class="flex items-center space-x-4">
          {#if enableAutoSave && !editMode}
            <Button type="button" variant="outline" on:on:on:click={() => { if (ondraft) ondraft({ data: $form }); }}>
              Save as Draft
            </Button>
          {/if}
        </div>

        <div class="flex items-center space-x-3">
          <Button type="button" variant="outline">
            Cancel
          </Button>
          
          <Button 
            type="submit" 
            disabled={$submitting || !$isValid}
            class="min-w-[120px]"
          >
            {#if $submitting}
              <Loader2 class="mr-2 h-4 w-4 animate-spin" />
              {editMode ? 'Updating...' : 'Creating...'}
            {:else}
              {editMode ? 'Update Case' : 'Create Case'}
            {/if}
          </Button>
        </div>
      </div>
    </form>
  </Card.Content>
</Card.Root>
{/if}

{#if componentError}
  <div class="error-boundary bg-red-50 border border-red-200 rounded-lg p-6 m-4">
    <h2 class="text-lg font-semibold text-red-800 mb-2">Form Error</h2>
    <p class="text-red-700 mb-4">The case form encountered an error:</p>
    <p class="text-red-600 font-mono text-sm mb-4 bg-red-100 p-2 rounded">{componentError.message}</p>
    <Button 
      on:on:on:click={() => { componentError = null; }}
      variant="outline"
      class="border-red-300 text-red-700 hover:bg-red-50"
    >
      Dismiss Error
    </Button>
  </div>
{/if}

<style lang="postcss">
  /* Custom styles for enhanced form interactions */
  .form-field-error {
    @apply border-destructive focus:ring-destructive;
  }
  
  .form-field-valid {
    @apply border-green-500 focus:ring-green-500;
  }
  
  .upload-zone:hover {
    @apply border-primary/50 bg-primary/5;
  }
</style>
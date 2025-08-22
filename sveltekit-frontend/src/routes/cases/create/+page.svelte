<!--
  Case Creation Page - SuperForms + Zod + Enhanced Actions
  Demonstrates complete form enhancement pattern
-->
<script lang="ts">
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { enhance } from '$app/forms';
  import { onMount } from 'svelte';
  import EnhancedCaseFormWithZod from '$lib/components/forms/EnhancedCaseFormWithZod.svelte';
  import { toast } from '$lib/components/ui/toast';
  import { Alert } from '$lib/components/ui/alert';
  import { Button } from '$lib/components/ui/button';
  import { ArrowLeft, Save, AlertCircle, CheckCircle } from 'lucide-svelte';
  import type { PageData } from './$types';
  import type { CaseForm } from '$lib/schemas/forms';

  export let data: PageData;
  
  let isSubmitting = false;
  let showSuccess = false;
  let errorMessage = '';

  // Handle form submission success
  function handleFormSuccess(event: CustomEvent<{ case: any }>) {
    const { case: newCase } = event.detail;
    
    showSuccess = true;
    toast.success(`Case ${newCase.caseNumber} created successfully!`);
    
    // Redirect to case view after 2 seconds
    setTimeout(() => {
      goto(`/cases/${newCase.id}`);
    }, 2000);
  }

  // Handle form submission error
  function handleFormError(event: CustomEvent<{ message: string }>) {
    errorMessage = event.detail.message;
    toast.error(event.detail.message);
  }

  // Handle draft save
  function handleDraftSave(event: CustomEvent<{ data: CaseForm }>) {
    toast.info('Draft saved successfully');
  }

  // Handle form submission
  function handleFormSubmit(event: CustomEvent<{ data: CaseForm }>) {
    isSubmitting = true;
    errorMessage = '';
  }

  // Check for success message from server
  onMount(() => {
    if ($page.form?.message?.type === 'success') {
      showSuccess = true;
      toast.success($page.form.message.text);
      
      // Redirect if we have a redirect URL
      if ($page.form.message.data?.redirectUrl) {
        setTimeout(() => {
          goto($page.form.message.data.redirectUrl);
        }, 2000);
      }
    } else if ($page.form?.message?.type === 'error') {
      errorMessage = $page.form.message.text;
      toast.error($page.form.message.text);
    }
  });

  // Auto-save functionality
  let autoSaveTimeout: NodeJS.Timeout;
  
  function scheduleAutoSave(formData: CaseForm) {
    clearTimeout(autoSaveTimeout);
    
    autoSaveTimeout = setTimeout(async () => {
      try {
        const response = await fetch('?/saveDraft', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(formData)
        });
        
        if (response.ok) {
          console.log('Auto-save successful');
        }
      } catch (error) {
        console.warn('Auto-save failed:', error);
      }
    }, 3000); // Auto-save after 3 seconds of inactivity
  }
</script>

<svelte:head>
  <title>{data.editMode ? 'Edit Case' : 'Create New Case'} - Legal AI Platform</title>
  <meta name="description" content={data.editMode ? 'Edit case information and evidence' : 'Create a new legal case with evidence upload and AI analysis'} />
</svelte:head>

<div class="container mx-auto px-4 py-8 max-w-6xl">
  <!-- Header -->
  <div class="mb-8">
    <div class="flex items-center justify-between">
      <div class="flex items-center space-x-4">
        <Button 
          variant="ghost" 
          size="sm"
          onclick={() => goto('/cases')}
          class="flex items-center space-x-2"
        >
          <ArrowLeft class="h-4 w-4" />
          <span>Back to Cases</span>
        </Button>
        
        <div class="h-6 border-l border-muted-foreground/20"></div>
        
        <div>
          <h1 class="text-3xl font-bold tracking-tight">
            {data.editMode ? 'Edit Case' : 'Create New Case'}
          </h1>
          <p class="text-muted-foreground mt-1">
            {data.editMode 
              ? 'Update case information and manage evidence' 
              : 'Enter case details, upload evidence, and enable AI analysis'
            }
          </p>
        </div>
      </div>

      <!-- Quick Actions -->
      <div class="flex items-center space-x-3">
        {#if !data.editMode}
          <Button 
            variant="outline" 
            onclick={() => goto('/cases/templates')}
            class="flex items-center space-x-2"
          >
            <Save class="h-4 w-4" />
            <span>Use Template</span>
          </Button>
        {/if}
      </div>
    </div>
  </div>

  <!-- Success Alert -->
  {#if showSuccess}
    <Alert.Root class="mb-6 border-green-200 bg-green-50">
      <CheckCircle class="h-4 w-4 text-green-600" />
      <Alert.Title class="text-green-800">Success!</Alert.Title>
      <Alert.Description class="text-green-700">
        Case has been {data.editMode ? 'updated' : 'created'} successfully. 
        {#if !data.editMode}
          Redirecting to case view...
        {/if}
      </Alert.Description>
    </Alert.Root>
  {/if}

  <!-- Error Alert -->
  {#if errorMessage}
    <Alert.Root variant="destructive" class="mb-6">
      <AlertCircle class="h-4 w-4" />
      <Alert.Title>Error</Alert.Title>
      <Alert.Description>{errorMessage}</Alert.Description>
    </Alert.Root>
  {/if}

  <!-- Form -->
  <EnhancedCaseFormWithZod
    data={data.form}
    submitAction={data.editMode ? `?/updateCase&id=${data.caseId}` : '?/createCase'}
    editMode={data.editMode}
    enableAutoSave={true}
    enableRealTimeValidation={true}
    onsubmit={handleFormSubmit}
    onsuccess={handleFormSuccess}
    onerror={handleFormError}
    ondraft={handleDraftSave}
  />

  <!-- Help Text -->
  <div class="mt-8 p-6 bg-muted/30 rounded-lg border">
    <h3 class="font-semibold mb-3 flex items-center space-x-2">
      <AlertCircle class="h-5 w-5 text-primary" />
      <span>Form Enhancement Features</span>
    </h3>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-muted-foreground">
      <div>
        <h4 class="font-medium text-foreground mb-2">SuperForms Integration</h4>
        <ul class="space-y-1">
          <li>• Real-time validation with Zod schemas</li>
          <li>• Progressive enhancement with JavaScript</li>
          <li>• Server-side validation fallback</li>
          <li>• Type-safe form handling</li>
        </ul>
      </div>
      
      <div>
        <h4 class="font-medium text-foreground mb-2">Enhanced Actions</h4>
        <ul class="space-y-1">
          <li>• File upload with progress tracking</li>
          <li>• Auto-save functionality</li>
          <li>• Error handling and recovery</li>
          <li>• Success state management</li>
        </ul>
      </div>
      
      <div>
        <h4 class="font-medium text-foreground mb-2">XState Integration</h4>
        <ul class="space-y-1">
          <li>• State machine driven form flow</li>
          <li>• Predictable state transitions</li>
          <li>• Complex form orchestration</li>
          <li>• Event-driven updates</li>
        </ul>
      </div>
      
      <div>
        <h4 class="font-medium text-foreground mb-2">Production Features</h4>
        <ul class="space-y-1">
          <li>• Comprehensive error handling</li>
          <li>• Audit trail logging</li>
          <li>• File validation and storage</li>
          <li>• Notification system integration</li>
        </ul>
      </div>
    </div>
  </div>
</div>

<style>
  /* Ensure smooth transitions for state changes */
  .container {
    @apply transition-all duration-200 ease-in-out;
  }
  
  /* Form success animation */
  :global(.form-success) {
    animation: slideInFromTop 0.3s ease-out;
  }
  
  @keyframes slideInFromTop {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>
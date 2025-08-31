<script lang="ts">
  import { $props } from 'svelte';
  import type { BitsDemoProps } from '$lib/types/component-props.js';

  // Component library choice - toggle between bits-ui and melt-ui
  import { Dialog, Button, Select, AlertDialog } from 'bits-ui';
  import MeltButton from '$lib/components/ui/MeltButton.svelte';
  import MeltDialog from '$lib/components/ui/MeltDialog.svelte';
  import MeltSelect from '$lib/components/ui/MeltSelect.svelte';
  import { fade } from 'svelte/transition';
  import { createToaster, melt } from 'melt';
  import { flip } from 'svelte/animate';
  import { fly } from 'svelte/transition';
  
  let { 
    caseTypes = [
      { value: 'criminal', label: 'Criminal Cases' },
      { value: 'civil', label: 'Civil Cases' },
      { value: 'family', label: 'Family Law' },
      { value: 'corporate', label: 'Corporate Law' }
    ],
    useLibrary = 'melt-ui',
    class: class,
    id,
    'data-testid': testId
  }: BitsDemoProps = $props();
  
  // Component selection based on library choice
  let SelectedButton = $derived(useLibrary === 'melt-ui' ? MeltButton : Button);
  let SelectedDialog = $derived(useLibrary === 'melt-ui' ? MeltDialog : Dialog);
  let SelectedSelect = $derived(useLibrary === 'melt-ui' ? MeltSelect : Select);
  
  interface ToastData {
    title?: string;
    description?: string;
    color: string;
  }

  let dialogOpen = false;
  let alertOpen = false;
  
  // Melt-UI Toast/Notification setup
  const {
    elements: { content, title, description, close },
    helpers: { addToast },
    states: { toasts },
    actions: { portal }
  } = createToaster<ToastData>();
  
  // Notification functions with actual API calls
  async function showSuccessNotification() {
    try {
      const response = await fetch('/api/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: 'Demo Case ' + Date.now(),
          description: 'Demonstration case created from BitsDemo component',
          priority: 'medium',
          status: 'open'
        })
      });

      if (response.ok) {
        const result = await response.json();
        addToast({
          data: {
            title: 'Case Created Successfully',
            description: `Case ${result.case?.caseNumber} created and saved to database.`,
            color: 'success',
          },
        });
      } else {
        throw new Error('Failed to create case');
      }
    } catch (error) {
      addToast({
        data: {
          title: 'Case Creation Failed',
          description: 'Unable to create case via API. Check backend connection.',
          color: 'error',
        },
      });
    }
  }

  async function showWarningNotification() {
    try {
      const response = await fetch('/api/comprehensive-integration', {
        method: 'GET'
      });

      if (response.ok) {
        const result = await response.json();
        addToast({
          data: {
            title: 'System Status Check',
            description: `Services: ${result.system_overview?.healthy_services || 0}/${result.system_overview?.total_services || 0} healthy`,
            color: 'warning',
          },
        });
      } else {
        throw new Error('Health check failed');
      }
    } catch (error) {
      addToast({
        data: {
          title: 'Health Check Failed',
          description: 'Unable to check system health. Backend may be down.',
          color: 'error',
        },
      });
    }
  }

  async function showErrorNotification() {
    try {
      const response = await fetch('/api/v1/upload?action=health');
      
      if (response.ok) {
        addToast({
          data: {
            title: 'Upload Service Test',
            description: 'Upload service is healthy and responding.',
            color: 'success',
          },
        });
      } else {
        throw new Error('Upload service unhealthy');
      }
    } catch (error) {
      addToast({
        data: {
          title: 'Upload Service Error',
          description: 'Upload service is not responding. Check backend services.',
          color: 'error',
        },
      });
    }
  }

  async function showInfoNotification() {
    try {
      const response = await fetch('/api/v1/quic/metrics', {
        method: 'GET'
      });
      
      if (response.ok) {
        const result = await response.json();
        addToast({
          data: {
            title: 'Multi-Protocol Check',
            description: `QUIC metrics available. P99: ${result.p99 || 'N/A'}ms`,
            color: 'info',
          },
        });
      } else {
        addToast({
          data: {
            title: 'Multi-Protocol Test',
            description: 'Testing REST, gRPC, QUIC protocol integration.',
            color: 'info',
          },
        });
      }
    } catch (error) {
      addToast({
        data: {
          title: 'Protocol Integration Test',
          description: 'Testing multi-protocol backend integration.',
          color: 'info',
        },
      });
    }
  }
</script>

<div class="mx-auto px-4 max-w-7xl">
  <h2 class="mx-auto px-4 max-w-7xl">Bits UI Components Demo</h2>
  
  <!-- Melt-UI Notification Demo Section -->
  <div class="mx-auto px-4 max-w-7xl">
    <h3 class="mx-auto px-4 max-w-7xl">Melt-UI Notifications Demo</h3>
    <div class="mx-auto px-4 max-w-7xl">
      <button class="mx-auto px-4 max-w-7xl" click={() => showSuccessNotification()}>
        Success Notification
      </button>
      <button class="mx-auto px-4 max-w-7xl" click={() => showWarningNotification()}>
        Warning Notification
      </button>
      <button class="mx-auto px-4 max-w-7xl" click={() => showErrorNotification()}>
        Error Notification
      </button>
      <button class="mx-auto px-4 max-w-7xl" click={() => showInfoNotification()}>
        Info Notification
      </button>
    </div>
  </div>
  
  <!-- Bits UI Button -->
  <Button.Root class="mx-auto px-4 max-w-7xl" on:click={showSuccessNotification}>
    Create New Case
  </Button.Root>
  
  <!-- Bits UI Select -->
  <div class="mx-auto px-4 max-w-7xl">
    <label class="mx-auto px-4 max-w-7xl" for="practice-area-select">Legal Practice Area</label>
    <SelectRoot type="single" on:valuechange={() => showWarningNotification()}>
      <SelectTrigger class="mx-auto px-4 max-w-7xl" id="practice-area-select">
        Select practice area...
      </SelectTrigger>
      <Select.Portal>
        <SelectContent class="mx-auto px-4 max-w-7xl">
          {#each caseTypes as type}
            <SelectItem value={type.value} class="mx-auto px-4 max-w-7xl">
              {type.label}
            </SelectItem>
          {/each}
        </SelectContent>
      </Select.Portal>
    </SelectRoot>
  </div>
  
  <!-- Bits UI Dialog -->
  <Dialog.Root bind:open={dialogOpen} on:openchange={(open) => { if (open) showInfoNotification(); }}>
    <Dialog.Trigger class="mx-auto px-4 max-w-7xl">
      Case Management Options
    </Dialog.Trigger>
    <Dialog.Portal>
      <Dialog.Overlay class="mx-auto px-4 max-w-7xl" />
      <Dialog.Content class="mx-auto px-4 max-w-7xl">
        <Dialog.Title class="mx-auto px-4 max-w-7xl">
          Case Management System
        </Dialog.Title>
        <Dialog.Description class="mx-auto px-4 max-w-7xl">
          Manage your legal cases with our comprehensive case management system. 
          Track evidence, deadlines, and case progress all in one place.
        </Dialog.Description>
        
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <h4>Evidence Management</h4>
            <p>Upload, organize and analyze case evidence</p>
          </div>
          <div class="mx-auto px-4 max-w-7xl">
            <h4>Timeline Tracking</h4>
            <p>Keep track of important dates and deadlines</p>
          </div>
          <div class="mx-auto px-4 max-w-7xl">
            <h4>AI Analysis</h4>
            <p>Get AI-powered insights on your cases</p>
          </div>
        </div>
        
        <div class="mx-auto px-4 max-w-7xl">
          <Dialog.Close class="mx-auto px-4 max-w-7xl">
            Close
          </Dialog.Close>
          <Button.Root class="mx-auto px-4 max-w-7xl">
            Get Started
          </Button.Root>
        </div>
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>
  
  <!-- Bits UI Alert Dialog -->
  <AlertDialog.Root bind:open={alertOpen}>
    <AlertDialog.Trigger class="mx-auto px-4 max-w-7xl">
      Delete Case
    </AlertDialog.Trigger>
    <AlertDialog.Portal>
      <AlertDialog.Overlay class="mx-auto px-4 max-w-7xl" />
      <AlertDialog.Content class="mx-auto px-4 max-w-7xl">
        <AlertDialog.Title class="mx-auto px-4 max-w-7xl">
          Delete Case Confirmation
        </AlertDialog.Title>
        <AlertDialog.Description class="mx-auto px-4 max-w-7xl">
          Are you sure you want to delete this case? This action cannot be undone and will permanently remove all case data, evidence, and related documents.
        </AlertDialog.Description>
        
        <div class="mx-auto px-4 max-w-7xl">
          <AlertDialog.Cancel class="mx-auto px-4 max-w-7xl">
            Cancel
          </AlertDialog.Cancel>
          <AlertDialog.Action class="mx-auto px-4 max-w-7xl" on:click={showErrorNotification}>
            Delete Permanently
          </AlertDialog.Action>
        </div>
      </AlertDialog.Content>
    </AlertDialog.Portal>
  </AlertDialog.Root>
  
  <div class="mx-auto px-4 max-w-7xl">
    <p class="mx-auto px-4 max-w-7xl">
      <strong>Demo:</strong> Bits UI components provide accessible, unstyled components.
      Melt-UI notifications provide toast/alert functionality.
    </p>
  </div>
</div>

<!-- Melt-UI Toast/Notification Container -->
<div class="mx-auto px-4 max-w-7xl" use:portal>
  {#each $toasts as { id, data } (id)}
    <div
      class="mx-auto px-4 max-w-7xl"
      animate:flip={{ duration: 500 }}
      in:fly={{ duration: 150, x: '100%' }}
      out:fly={{ duration: 150, x: '100%'  }}
      use:melt={$content(id)}
    >
      <div class="mx-auto px-4 max-w-7xl">
        {#if (data as ToastData).title}
          <div class="mx-auto px-4 max-w-7xl" use:melt={$title(id)}>
            {(data as ToastData).title}
          </div>
        {/if}
        <button class="mx-auto px-4 max-w-7xl" use:melt={$close(id)} aria-label="Close notification">
          ✕
        </button>
      </div>
      {#if (data as ToastData).description}
        <div class="mx-auto px-4 max-w-7xl" use:melt={$description(id)}>
          {(data as ToastData).description}
        </div>
      {/if}
    </div>
  {/each}
</div>

<style>
  .bits-demo {
    max-width: 600px;
    margin: 0 auto;
    padding: var(--spacing-lg);
  }

  /* Notification Demo Styles */
  .notification-demo {
    background-color: var(--color-surface);
    border-color: var(--color-border);
  }

  .notification-buttons {
    display: flex;
    gap: var(--spacing-sm);
    flex-wrap: wrap;
  }

  .btn {
    padding: var(--spacing-sm) var(--spacing-md);
    border: none;
    border-radius: var(--radius-md);
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
    font-size: var(--font-size-sm);
  }

  .btn:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
  }

  .btn-success {
    background-color: #10b981;
    color: white;
  }

  .btn-success:hover {
    background-color: #059669;
  }

  .btn-warning {
    background-color: #f59e0b;
    color: white;
  }

  .btn-warning:hover {
    background-color: #d97706;
  }

  .btn-danger {
    background-color: #ef4444;
    color: white;
  }

  .btn-danger:hover {
    background-color: #dc2626;
  }

  .btn-info {
    background-color: #3b82f6;
    color: white;
  }

  .btn-info:hover {
    background-color: #2563eb;
  }

  /* Toast/Notification Styles */
  .toast-container {
    position: fixed;
    top: var(--spacing-lg);
    right: var(--spacing-lg);
    z-index: 100;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
    max-width: 400px;
  }

  .toast {
    background-color: var(--color-background);
    border: 1px solid;
    border-radius: var(--radius-lg);
    padding: var(--spacing-md);
    box-shadow: var(--shadow-lg);
    min-width: 300px;
    position: relative;
  }

  .toast-success {
    border-color: #10b981;
    background-color: #ecfdf5;
  }

  .toast-warning {
    border-color: #f59e0b;
    background-color: #fffbeb;
  }

  .toast-error {
    border-color: #ef4444;
    background-color: #fef2f2;
  }

  .toast-info {
    border-color: #3b82f6;
    background-color: #eff6ff;
  }

  .toast-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: var(--spacing-xs);
  }

  .toast-title {
    font-weight: 600;
    font-size: var(--font-size-sm);
    color: var(--color-text);
    margin-right: var(--spacing-sm);
  }

  .toast-description {
    font-size: var(--font-size-sm);
    color: var(--color-text-muted);
    line-height: 1.4;
  }

  .toast-close {
    background: none;
    border: none;
    font-size: var(--font-size-sm);
    color: var(--color-text-muted);
    cursor: pointer;
    padding: 0;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    transition: all var(--transition-fast);
    flex-shrink: 0;
  }

  .toast-close:hover {
    background-color: var(--color-surface);
    color: var(--color-text);
  }

  :global(.select-content) {
    background-color: var(--color-background);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-lg);
    padding: var(--spacing-xs);
    z-index: 50;
    max-height: 200px;
    overflow-y: auto;
  }

  :global(.select-item) {
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background-color var(--transition-fast);
    display: block;
    width: 100%;
    text-align: left;
  }

  :global(.select-item:hover),
  :global(.select-item[data-highlighted]) {
    background-color: var(--color-surface);
  }

  :global(.dialog-overlay) {
    position: fixed;
    inset: 0;
    z-index: 50;
    background-color: rgb(0 0 0 / 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-lg);
  }

  :global(.dialog-content) {
    background-color: var(--color-background);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    padding: var(--spacing-xl);
    max-width: 500px;
    width: 100%;
    max-height: 90vh;
    overflow-y: auto;
    position: relative;
  }

  :global(.dialog-title) {
    font-size: var(--font-size-xl);
    font-weight: 600;
    margin-bottom: var(--spacing-md);
    color: var(--color-text);
  }

  :global(.dialog-description) {
    color: var(--color-text-muted);
    margin-bottom: var(--spacing-lg);
    line-height: 1.6;
  }

  .case-options {
    margin-bottom: var(--spacing-lg);
  }

  .case-option {
    padding: var(--spacing-md);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    margin-bottom: var(--spacing-sm);
    transition: all var(--transition-fast);
  }

  .case-option:hover {
    border-color: var(--color-primary);
    box-shadow: var(--shadow-sm);
  }

  .case-option h4 {
    margin: 0 0 var(--spacing-xs) 0;
    font-weight: 600;
    color: var(--color-text);
  }

  .case-option p {
    margin: 0;
    font-size: var(--font-size-sm);
    color: var(--color-text-muted);
  }

  .dialog-actions,
  .alert-actions {
    display: flex;
    gap: var(--spacing-sm);
    justify-content: flex-end;
  }

  :global(.text-danger) {
    color: var(--color-danger);
  }

  .text-muted {
    color: var(--color-text-muted);
  }

  .border {
    border: 1px solid var(--color-border);
  }

  .rounded {
    border-radius: var(--radius-md);
  }
</style>
<!-- TODO: migrate export lets to $props(); CommonProps assumed. -->

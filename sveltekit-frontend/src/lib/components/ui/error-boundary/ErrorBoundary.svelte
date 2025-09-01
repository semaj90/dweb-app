<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { Button } from 'bits-ui';
  import { Card } from 'bits-ui';
  import ModernButton from '$lib/components/ui/button/Button.svelte';
  
  interface Props {
    fallback?: any;
    children: any;
    onError?: (error: Error, errorInfo?: any) => void;
  }
  
  let { fallback, children, onError }: Props = $props();
  
  let hasError = $state(false);
  let error = $state<Error | null>(null);
  let errorId = $state<string>('');
  
  // Error logging
  function logError(err: Error, context?: any) {
    const errorData = {
      id: errorId,
      message: err.message,
      stack: err.stack,
      url: $page.url.pathname,
      timestamp: new Date().toISOString(),
      userAgent: globalThis.navigator?.userAgent,
      context
    };
    
    console.error('YoRHa Error Boundary:', errorData);
    
    // In production, send to error tracking service
    // fetch('/api/errors', { method: 'POST', body: JSON.stringify(errorData) });
    
    onError?.(err, errorData);
  }
  
  // Global error handler
  function handleGlobalError(event: ErrorEvent) {
    if (!hasError) {
      const err = new Error(event.message);
      err.stack = `${event.filename}:${event.lineno}:${event.colno}`;
      
      hasError = true;
      error = err;
      errorId = `ERR_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      logError(err, { type: 'global', event });
    }
  }
  
  // Unhandled promise rejection handler
  function handleUnhandledRejection(event: PromiseRejectionEvent) {
    if (!hasError) {
      const err = new Error(event.reason?.message || 'Unhandled promise rejection');
      
      hasError = true;
      error = err;
      errorId = `ERR_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      logError(err, { type: 'unhandled_rejection', reason: event.reason });
    }
  }
  
  // Reset error state
  function resetError() {
    hasError = false;
    error = null;
    errorId = '';
  }
  
  // Reload page
  function reloadPage() {
    globalThis.location?.reload();
  }
  
  onMount(() => {
    // Add global error listeners
    globalThis.addEventListener('error', handleGlobalError);
    globalThis.addEventListener('unhandledrejection', handleUnhandledRejection);
    
    return () => {
      globalThis.removeEventListener('error', handleGlobalError);
      globalThis.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  });
</script>

{#if hasError && error}
  <!-- Error State -->
  <div class="min-h-screen bg-nier-bg-primary text-nier-text-primary flex items-center justify-center p-golden-lg">
    <Card.Root class="bg-nier-bg-secondary border-red-500/30 max-w-2xl w-full">
      <Card.Header class="text-center pb-golden-lg">
        <div class="mb-golden-md">
          <!-- YoRHa Error Icon -->
          <div class="w-16 h-16 mx-auto bg-red-500/20 rounded-full flex items-center justify-center">
            <svg class="w-8 h-8 text-red-400" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
          </div>
        </div>
        <Card.Title class="text-2xl font-bold text-red-400 uppercase tracking-wide mb-golden-sm">
          System Error Detected
        </Card.Title>
        <Card.Description class="text-nier-text-secondary">
          The YoRHa Legal AI system encountered an unexpected error. Our androids are investigating the issue.
        </Card.Description>
      </Card.Header>
      
      <Card.Content class="space-y-golden-lg">
        <!-- Error Details -->
        <div class="bg-nier-bg-tertiary border border-nier-border-muted rounded p-golden-md">
          <h3 class="text-sm font-bold text-nier-accent-warm uppercase tracking-wide mb-golden-sm">
            Error Details
          </h3>
          <div class="font-mono text-sm space-y-golden-xs">
            <div class="grid grid-cols-4 gap-2">
              <span class="text-nier-text-secondary">ID:</span>
              <span class="col-span-3 text-red-400">{errorId}</span>
              
              <span class="text-nier-text-secondary">Message:</span>
              <span class="col-span-3 text-nier-text-primary">{error.message}</span>
              
              <span class="text-nier-text-secondary">Location:</span>
              <span class="col-span-3 text-nier-text-primary">{$page.url.pathname}</span>
              
              <span class="text-nier-text-secondary">Time:</span>
              <span class="col-span-3 text-nier-text-primary">{new Date().toLocaleString()}</span>
            </div>
          </div>
        </div>
        
        <!-- Stack Trace (Development) -->
        {#if error.stack && globalThis.location?.hostname === 'localhost'}
          <details class="bg-nier-bg-tertiary border border-nier-border-muted rounded p-golden-md">
            <summary class="text-sm font-bold text-nier-text-secondary uppercase tracking-wide cursor-pointer hover:text-nier-accent-warm transition-colors">
              Stack Trace (Development)
            </summary>
            <pre class="font-mono text-xs text-nier-text-primary mt-golden-sm overflow-x-auto whitespace-pre-wrap">{error.stack}</pre>
          </details>
        {/if}
        
        <!-- Action Buttons -->
        <div class="flex flex-col sm:flex-row gap-golden-sm justify-center">
          <ModernButton 
            on:on:on:click={resetError}
            variant="primary"
            class="bg-gradient-to-r from-nier-accent-warm to-nier-accent-cool text-nier-bg-primary"
          >
            Try Again
          </ModernButton>
          
          <ModernButton 
            on:on:on:click={reloadPage}
            variant="outline"
            class="border-nier-accent-cool text-nier-accent-cool hover:bg-nier-accent-cool hover:text-nier-bg-primary"
          >
            Reload Page
          </ModernButton>
          
          <ModernButton 
            href="/"
            variant="ghost"
            class="text-nier-text-secondary hover:text-nier-accent-warm hover:bg-nier-bg-tertiary"
          >
            Go Home
          </ModernButton>
        </div>
        
        <!-- Support Information -->
        <div class="text-center pt-golden-lg border-t border-nier-border-muted">
          <p class="text-xs text-nier-text-muted">
            If this error persists, please contact the YoRHa support team with error ID: 
            <code class="text-red-400 font-mono">{errorId}</code>
          </p>
        </div>
      </Card.Content>
    </Card.Root>
  </div>
{:else if fallback}
  {@render fallback()}
{:else}
  {@render children()}
{/if}

<style>
  /* Ensure error boundary styles don't interfere with global styles */
  details summary::-webkit-details-marker {
    display: none;
  }
  
  details summary::before {
    content: '▶';
    margin-right: 0.5rem;
    transition: transform 0.2s ease;
  }
  
  details[open] summary::before {
    transform: rotate(90deg);
  }
</style>
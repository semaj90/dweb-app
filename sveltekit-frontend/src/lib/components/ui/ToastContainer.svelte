<!-- Toast Container with NES.css Styling -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { toastService, type Toast } from '$lib/services/toast-service';
  import { CheckCircle, AlertCircle, AlertTriangle, Info, Upload, X, RotateCcw } from 'lucide-svelte';

  let toasts = $state<Toast[]>([]);
  let unsubscribe: (() => void) | null = null;

  onMount(() => {
    unsubscribe = toastService.subscribe((newToasts) => {
      toasts = newToasts;
    });
  });

  onDestroy(() => {
    unsubscribe?.();
  });

  function getToastIcon(type: Toast['type']) {
    switch (type) {
      case 'success': return CheckCircle;
      case 'error': return AlertCircle;
      case 'warning': return AlertTriangle;
      case 'info': return Info;
      case 'upload': return Upload;
      default: return Info;
    }
  }

  function getToastClass(type: Toast['type']): string {
    switch (type) {
      case 'success': return 'is-success';
      case 'error': return 'is-error';
      case 'warning': return 'is-warning';
      case 'info': return 'is-primary';
      case 'upload': return 'is-dark';
      default: return 'is-primary';
    }
  }

  function dismissToast(id: string) {
    toastService.dismiss(id);
  }

  function executeAction(action: NonNullable<Toast['actions']>[0]) {
    action.action();
  }
</script>

<!-- Toast Container -->
<div class="toast-container" role="region" aria-label="Notifications">
  {#each toasts as toast (toast.id)}
    <div 
      class="toast-item nes-container {getToastClass(toast.type)}"
      role="alert"
      aria-live={toast.type === 'error' ? 'assertive' : 'polite'}
      aria-describedby="toast-{toast.id}"
    >
      <!-- Toast Header -->
      <div class="toast-header">
        <div class="toast-icon">
          {#if getToastIcon(toast.type)}
            <svelte:component this={getToastIcon(toast.type)} size={16} />
          {/if}
        </div>
        <div class="toast-title">
          <span class="nes-text">{toast.title}</span>
        </div>
        {#if toast.dismissible}
          <button 
            type="button"
            class="toast-dismiss nes-btn is-error"
            onclick={() => dismissToast(toast.id)}
            aria-label="Dismiss notification"
          >
            <X size={12} />
          </button>
        {/if}
      </div>

      <!-- Toast Content -->
      <div class="toast-content" id="toast-{toast.id}">
        <p class="toast-message">{toast.message}</p>

        <!-- Progress Bar for Upload Toasts -->
        {#if toast.type === 'upload' && toast.progress !== undefined}
          <div class="toast-progress">
            <div class="nes-container is-rounded progress-container">
              <progress 
                class="nes-progress {toast.progress < 100 ? 'is-primary' : 'is-success'}" 
                value={toast.progress} 
                max="100"
                aria-label="Upload progress"
              ></progress>
              <span class="progress-text">{Math.round(toast.progress)}%</span>
            </div>
          </div>
        {/if}

        <!-- Toast Actions -->
        {#if toast.actions && toast.actions.length > 0}
          <div class="toast-actions">
            {#each toast.actions as action}
              <button 
                type="button"
                class="nes-btn {action.style === 'primary' ? 'is-primary' : action.style === 'danger' ? 'is-error' : ''}"
                onclick={() => executeAction(action)}
              >
                {#if action.label === 'Retry'}
                  <RotateCcw size={12} />
                {/if}
                {action.label}
              </button>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Toast Timestamp -->
      <div class="toast-timestamp">
        <span class="nes-text is-disabled">
          {toast.createdAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  {/each}
</div>

<style>
  .toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-width: 420px;
    width: 100%;
    pointer-events: none;
  }

  .toast-item {
    pointer-events: auto;
    background: #ffffff;
    border: 4px solid #212529;
    box-shadow: 4px 4px 0px rgba(0, 0, 0, 0.3);
    animation: slideInRight 0.3s ease-out;
    position: relative;
    padding: 16px;
    font-family: "Press Start 2P", cursive;
  }

  .toast-item.is-success {
    border-color: #92cc41;
    background: #f8fff8;
  }

  .toast-item.is-error {
    border-color: #e76e55;
    background: #fff8f8;
  }

  .toast-item.is-warning {
    border-color: #f7d51d;
    background: #fffef8;
  }

  .toast-item.is-primary {
    border-color: #209cee;
    background: #f8fcff;
  }

  .toast-item.is-dark {
    border-color: #212529;
    background: #f5f5f5;
  }

  .toast-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .toast-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
  }

  .toast-title {
    flex: 1;
    font-size: 10px;
    font-weight: bold;
  }

  .toast-dismiss {
    padding: 4px;
    font-size: 8px;
    line-height: 1;
    min-width: auto;
    height: auto;
  }

  .toast-content {
    margin-bottom: 8px;
  }

  .toast-message {
    font-size: 8px;
    line-height: 1.4;
    margin: 0;
    word-wrap: break-word;
  }

  .toast-progress {
    margin-top: 8px;
  }

  .progress-container {
    position: relative;
    padding: 8px;
    background: rgba(255, 255, 255, 0.5);
    border: 2px solid #212529;
  }

  .progress-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 8px;
    font-weight: bold;
    color: #212529;
    text-shadow: 1px 1px 0px rgba(255, 255, 255, 0.8);
  }

  .toast-actions {
    display: flex;
    gap: 8px;
    margin-top: 8px;
    flex-wrap: wrap;
  }

  .toast-actions .nes-btn {
    font-size: 8px;
    padding: 8px 12px;
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .toast-timestamp {
    position: absolute;
    bottom: 4px;
    right: 8px;
    font-size: 6px;
  }

  /* Animations */
  @keyframes slideInRight {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }

  .toast-item.removing {
    animation: slideOutRight 0.3s ease-in forwards;
  }

  @keyframes slideOutRight {
    from {
      transform: translateX(0);
      opacity: 1;
    }
    to {
      transform: translateX(100%);
      opacity: 0;
    }
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .toast-container {
      top: 10px;
      right: 10px;
      left: 10px;
      max-width: none;
    }

    .toast-item {
      padding: 12px;
    }

    .toast-title {
      font-size: 9px;
    }

    .toast-message {
      font-size: 7px;
    }

    .toast-actions .nes-btn {
      font-size: 7px;
      padding: 6px 10px;
    }
  }

  /* Progress bar custom styling */
  .nes-progress {
    position: relative;
    height: 20px;
  }

  /* Hide default progress styling and add pixel styling */
  .nes-progress::-webkit-progress-bar {
    background-color: #e5e7eb;
    border: 1px solid #212529;
  }

  .nes-progress::-webkit-progress-value {
    background-color: #92cc41;
    border-right: 2px solid #212529;
  }

  .nes-progress.is-primary::-webkit-progress-value {
    background-color: #209cee;
  }

  .nes-progress.is-success::-webkit-progress-value {
    background-color: #92cc41;
  }

  /* Firefox */
  .nes-progress::-moz-progress-bar {
    background-color: #92cc41;
    border: 1px solid #212529;
  }

  .nes-progress.is-primary::-moz-progress-bar {
    background-color: #209cee;
  }
</style>
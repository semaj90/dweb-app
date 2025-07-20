<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let show: boolean = false;
  export let title: string = '';

  const dispatch = createEventDispatcher();

  function close() {
    show = false;
    dispatch('close');
  }
</script>

{#if show}
  <div class="modal-backdrop" on:click={close}></div>
  <div class="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="modal-title">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="modal-title">{title}</h5>
        <button type="button" class="btn-close" aria-label="Close" on:click={close}></button>
      </div>
      <div class="modal-body">
        <slot></slot>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 1040;
  }

  .modal-dialog {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1050;
    width: 90%; /* Adjust as needed */
    max-width: 800px; /* Max width for larger screens */
    background: white;
    border-radius: 8px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    display: flex;
    flex-direction: column;
  }

  .modal-content {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    border-bottom: 1px solid #dee2e6;
  }

  .modal-title {
    margin-bottom: 0;
    line-height: 1.5;
  }

  .btn-close {
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0.5rem;
  }

  .modal-body {
    padding: 1rem;
    overflow-y: auto; /* Enable scrolling for content */
    flex-grow: 1; /* Allow body to take available space */
  }
</style>

<div
  role="button"
  tabindex="0"
  on:click={close}
  on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') close(); }}
  aria-label="Close modal"
>
  <!-- Modal content here -->
</div>
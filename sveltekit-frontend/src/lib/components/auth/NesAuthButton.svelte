<!--
  Standalone NES Auth Button Component
  Triggers the NES modal from anywhere in your app
-->
<script lang="ts">
  import NesAuthModal from './NesAuthModal.svelte';

  interface Props {
    text?: string;
    variant?: 'primary' | 'warning' | 'success' | 'error';
    size?: 'small' | 'medium' | 'large';
    icon?: string;
    form?: any;
  }

  let {
    text = 'Retro Auth',
    variant = 'primary',
    size = 'medium',
    icon = '🎮',
    form
  }: Props = $props();

  let isModalOpen = $state(false);

  function openModal() {
    isModalOpen = true;
  }

  function closeModal() {
    isModalOpen = false;
  }

  // Dynamic classes based on props
  let buttonClasses = $derived(() => {
    let classes = 'nes-btn';

    switch (variant) {
      case 'warning':
        classes += ' is-warning';
        break;
      case 'success':
        classes += ' is-success';
        break;
      case 'error':
        classes += ' is-error';
        break;
      default:
        classes += ' is-primary';
    }

    switch (size) {
      case 'small':
        classes += ' nes-btn-small';
        break;
      case 'large':
        classes += ' nes-btn-large';
        break;
    }

    return classes;
  });
</script>

<svelte:head>
  <!-- Import nes.css for the button styling -->
  <link href="https://unpkg.com/nes.css@latest/css/nes.min.css" rel="stylesheet" />
  <link href="https://fonts.googleapis.com/css?family=Press+Start+2P" rel="stylesheet">
</svelte:head>

<button
  type="button"
  class={buttonClasses}
  on:onclick={openModal}
>
  {#if icon}{icon} {/if}{text}
</button>

<NesAuthModal
  bind:isOpen={isModalOpen}
  {form}
  close={closeModal}
/>

<style>
  /* Custom button size classes */
  :global(.nes-btn-small) {
    font-size: 0.6rem;
    padding: 0.5rem 1rem;
  }

  :global(.nes-btn-large) {
    font-size: 1rem;
    padding: 1rem 2rem;
  }

  /* Button hover effects */
  :global(.nes-btn) {
    transition: transform 0.1s ease;
  }

  :global(.nes-btn:hover) {
    transform: scale(1.02);
  }

  :global(.nes-btn:active) {
    transform: scale(0.98);
  }
</style>
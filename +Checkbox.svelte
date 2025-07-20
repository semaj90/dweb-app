<script lang="ts">
  export let checked: boolean = false;
  export let label: string = '';
  export let id: string = '';

  // Generate a unique ID if not provided
  if (!id) {
    id = `checkbox-${Math.random().toString(36).substr(2, 9)}`;
  }
</script>

<div class="checkbox-container">
  <input type="checkbox" id={id} bind:checked={checked} class="checkbox-input" />
  {#if label}
    <label for={id} class="checkbox-label">{label}</label>
  {/if}
</div>

<style>
  .checkbox-container {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .checkbox-input {
    /* Hide default checkbox */
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
    width: 1.25rem;
    height: 1.25rem;
    border: 2px solid #ccc;
    border-radius: 0.25rem;
    cursor: pointer;
    position: relative;
    outline: none;
    transition: all 0.2s ease-in-out;
  }

  .checkbox-input:checked {
    background-color: #007bff;
    border-color: #007bff;
  }

  .checkbox-input:checked::after {
    content: '';
    position: absolute;
    top: 0.1875rem;
    left: 0.1875rem;
    width: 0.625rem;
    height: 0.3125rem;
    border: 2px solid #fff;
    border-top: none;
    border-right: none;
    transform: rotate(-45deg);
  }

  .checkbox-input:focus {
    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
  }

  .checkbox-label {
    margin-left: 0.5rem;
    cursor: pointer;
    color: #333;
  }
</style>
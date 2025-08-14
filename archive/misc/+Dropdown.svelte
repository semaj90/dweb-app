<script lang="ts">
  export let options: { value: string; label: string }[] = [];
  export let selected: string = '';
  export let placeholder: string = 'Select an option';
  export let label: string = '';
  export let id: string = ''; // Add id prop
</script>

<div class="dropdown-container">
  {#if label}
    <label for="dropdown-{label.replace(/\s+/g, '-')}" class="dropdown-label">{label}</label>
  {/if}
  <select id={id || `dropdown-${label.replace(/\s+/g, '-')}`} bind:value={selected} class="dropdown-select">
    {#if placeholder}
      <option value="" disabled selected>{placeholder}</option>
    {/if}
    {#each options as option}
      <option value={option.value}>{option.label}</option>
    {/each}
  </select>
</div>

<style>
  .dropdown-container {
    margin-bottom: 1rem;
  }

  .dropdown-label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: bold;
    color: #333;
  }

  .dropdown-select {
    width: 100%;
    padding: 0.75rem 1rem;
    border: 1px solid #ccc;
    border-radius: 0.375rem;
    background-color: #fff;
    font-size: 1rem;
    line-height: 1.5;
    color: #333;
    appearance: none; /* Remove default browser styling */
    -webkit-appearance: none;
    -moz-appearance: none;
    background-image: url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2020%2020%22%20fill%3D%22%23333%22%3E%3Cpath%20fill-rule%3D%22evenodd%22%20d%3D%22M5.293%207.293a1%201%200%20011.414%200L10%2010.586l3.293-3.293a1%201%200%20111.414%201.414l-4%204a1%201%200%2001-1.414%200l-4-4a1%201%200%20010-1.414z%22%20clip-rule%3D%22evenodd%22%2F%3E%3C%2Fsvg%3E');
    background-repeat: no-repeat;
    background-position: right 0.75rem center;
    background-size: 1.25rem;
  }

  .dropdown-select:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
  }
</style>
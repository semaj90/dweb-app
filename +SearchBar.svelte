<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import debounce from 'lodash-es/debounce';

  export let placeholder = 'Search...';
  export let value = '';

  const dispatch = createEventDispatcher<{
    search: string;
  }>();

  const debouncedSearch = debounce((searchTerm: string) => {
    dispatch('search', searchTerm);
  }, 300);

  $: if (value !== undefined) {
    debouncedSearch(value);
  }
</script>

<div class="search-container">
  <input
    type="text"
    {placeholder}
    bind:value
    class="search-input"
  />
  <svg xmlns="http://www.w3.org/2000/svg" class="search-icon" viewBox="0 0 20 20" fill="currentColor">
    <path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clip-rule="evenodd" />
  </svg>
</div>

<style>
  .search-container {
    position: relative;
    width: 100%;
    max-width: 500px;
  }

  .search-input {
    width: 100%;
    padding: 0.75rem 1rem;
    padding-left: 2.5rem;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 1rem;
  }

  .search-icon {
    position: absolute;
    left: 0.75rem;
    top: 50%;
    transform: translateY(-50%);
    width: 1.25rem;
    height: 1.25rem;
    color: #666;
  }
</style>

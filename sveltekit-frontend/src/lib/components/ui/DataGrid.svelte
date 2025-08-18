<script lang="ts">
  import type { Props } from "$lib/types/global";
  import { createEventDispatcher } from 'svelte';
  import { cn } from '$lib/utils/cn';
  import { ChevronDown, ChevronUp, MoreHorizontal, Search, Filter } from 'lucide-svelte';




  let {
    columns,
    data = [],
    loading = false,
    selectable = false,
    multiSelect = false,
    sortable = true,
    filterable = true,
    className = '',
    emptyMessage = 'No data available',
    children
  }: Props = $props();

  const dispatch = createEventDispatcher();
  
  let selectedRows = $state<Set<string | number>>(new Set());
  let sortConfig = $state<{column: string, direction: 'asc' | 'desc'} | null>(null);
  let searchQuery = $state('');
  let columnFilters = $state<Map<string, string>>(new Map());

  const filteredData = $derived(() => {
    let filtered = data;
    
    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(row =>
        columns.some(col => 
          String(row[col.key] || '').toLowerCase().includes(query)
        )
      );
    }
    
    // Apply column filters
    for (const [column, filter] of columnFilters) {
      if (filter.trim()) {
        filtered = filtered.filter(row =>
          String(row[column] || '').toLowerCase().includes(filter.toLowerCase())
        );
      }
    }
    
    return filtered;
  });

  const sortedData = $derived(() => {
    if (!sortConfig) return filteredData;
    
    return [...filteredData].sort((a, b) => {
      const aVal = a[sortConfig.column];
      const bVal = b[sortConfig.column];
      
      if (aVal === bVal) return 0;
      
      const result = aVal < bVal ? -1 : 1;
      return sortConfig.direction === 'desc' ? -result : result;
    });
  });

  function handleSort(column: string) {
    if (!sortable) return;
    
    if (sortConfig?.column === column) {
      sortConfig = {
        column,
        direction: sortConfig.direction === 'asc' ? 'desc' : 'asc'
      };
    } else {
      sortConfig = { column, direction: 'asc' };
    }
  }

  function handleRowSelect(rowId: string | number) {
    if (!selectable) return;
    
    if (multiSelect) {
      const newSelection = new Set(selectedRows);
      if (newSelection.has(rowId)) {
        newSelection.delete(rowId);
      } else {
        newSelection.add(rowId);
      }
      selectedRows = newSelection;
    } else {
      selectedRows = new Set([rowId]);
    }
    
    dispatch('selectionChange', { selectedRows: Array.from(selectedRows) });
  }

  function handleSelectAll() {
    if (!multiSelect) return;
    
    if (selectedRows.size === sortedData.length) {
      selectedRows = new Set();
    } else {
      selectedRows = new Set(sortedData.map(row => row.id));
    }
    
    dispatch('selectionChange', { selectedRows: Array.from(selectedRows) });
  }

  function handleColumnFilter(column: string, value: string) {
    if (value.trim()) {
      columnFilters.set(column, value);
    } else {
      columnFilters.delete(column);
    }
    columnFilters = new Map(columnFilters);
  }
</script>

<div class={cn('modern-data-grid', className)}>
  <!-- Header with search and filters -->
  {#if filterable}
    <div class="grid-toolbar">
      <div class="search-container">
        <Search class="search-icon" />
        <input
          type="text"
          placeholder="Search all columns..."
          bind:value={searchQuery}
          class="search-input"
        />
      </div>
      
      <div class="filter-actions">
        <button class="filter-button">
          <Filter class="h-4 w-4" />
          Filters
        </button>
      </div>
    </div>
  {/if}

  <!-- Data table -->
  <div class="table-container">
    <table class="data-table">
      <thead class="table-header">
        <tr class="header-row">
          {#if selectable && multiSelect}
            <th class="select-header">
              <input
                type="checkbox"
                checked={selectedRows.size === sortedData.length && sortedData.length > 0}
                onchange={handleSelectAll}
                class="checkbox-input"
              />
            </th>
          {/if}
          
          {#each columns as column}
            <th class="header-cell">
              <button
                class="header-button"
                onclick={() => handleSort(column.key)}
                disabled={!sortable || !column.sortable}
              >
                <span class="header-text">{column.title}</span>
                {#if sortable && column.sortable}
                  <div class="sort-icons">
                    {#if sortConfig?.column === column.key}
                      {#if sortConfig.direction === 'asc'}
                        <ChevronUp class="h-4 w-4 text-blue-600" />
                      {:else}
                        <ChevronDown class="h-4 w-4 text-blue-600" />
                      {/if}
                    {:else}
                      <ChevronDown class="h-4 w-4 text-gray-400" />
                    {/if}
                  </div>
                {/if}
              </button>
            </th>
          {/each}
          
          <th class="actions-header">
            <MoreHorizontal class="h-4 w-4 text-gray-400" />
          </th>
        </tr>
      </thead>
      
      <tbody class="table-body">
        {#if loading}
          <tr>
            <td colspan={columns.length + (selectable && multiSelect ? 1 : 0) + 1} class="loading-cell">
              <div class="loading-content">
                <div class="loading-spinner"></div>
                <span class="loading-text">Loading data...</span>
              </div>
            </td>
          </tr>
        {:else if sortedData.length === 0}
          <tr>
            <td colspan={columns.length + (selectable && multiSelect ? 1 : 0) + 1} class="empty-cell">
              <div class="empty-content">
                <div class="empty-icon">📄</div>
                <span class="empty-text">{emptyMessage}</span>
              </div>
            </td>
          </tr>
        {:else}
          {#each sortedData as row, index}
            <tr 
              class={cn('data-row', {
                'row-selected': selectedRows.has(row.id),
                'row-even': index % 2 === 0,
                'row-clickable': selectable
              })}
              onclick={() => handleRowSelect(row.id)}
            >
              {#if selectable && multiSelect}
                <td class="select-cell">
                  <input
                    type="checkbox"
                    checked={selectedRows.has(row.id)}
                    onchange={() => handleRowSelect(row.id)}
                    class="checkbox-input"
                  />
                </td>
              {/if}
              
              {#each columns as column}
                <td class="data-cell">
                  <div class="cell-content">
                    {#if column.formatter}
                      {column.formatter(row[column.key], row)}
                    {:else}
                      {row[column.key] || '—'}
                    {/if}
                  </div>
                </td>
              {/each}
              
              <td class="actions-cell">
                {@render children?.({ row, index })}
              </td>
            </tr>
          {/each}
        {/if}
      </tbody>
    </table>
  </div>
</div>

<style>
  .modern-data-grid {
    background-color: white;
    border: 1px solid rgb(229 231 235);
    border-radius: 12px;
    position: relative;
    overflow: hidden;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
  }

  .grid-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid rgb(243 244 246);
    background-color: rgb(249 250 251);
  }

  .search-container {
    position: relative;
    flex: 1;
    max-width: 24rem;
  }

  .search-icon {
    position: absolute;
    left: 0.75rem;
    top: 50%;
    transform: translateY(-50%);
    width: 1rem;
    height: 1rem;
    color: rgb(156 163 175);
  }

  .search-input {
    width: 100%;
    padding: 0.5rem 0.75rem 0.5rem 2.5rem;
    border: 1px solid rgb(209 213 219);
    border-radius: 0.5rem;
    font-size: 0.875rem;
    background-color: white;
    transition: border-color 0.15s;
  }

  .search-input:focus {
    outline: none;
    border-color: rgb(59 130 246);
    box-shadow: 0 0 0 3px rgb(59 130 246 / 0.1);
  }

  .filter-actions {
    display: flex;
    gap: 0.5rem;
  }

  .filter-button {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background-color: white;
    border: 1px solid rgb(209 213 219);
    border-radius: 0.5rem;
    font-size: 0.875rem;
    font-weight: 500;
    color: rgb(55 65 81);
    transition: all 0.15s;
  }

  .filter-button:hover {
    background-color: rgb(249 250 251);
    border-color: rgb(156 163 175);
  }

  .table-container {
    overflow: auto;
    max-height: 70vh;
  }

  .data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
  }

  .table-header {
    background-color: rgb(249 250 251);
    position: sticky;
    top: 0;
    z-index: 10;
  }

  .header-row {
    border-bottom: 1px solid rgb(229 231 235);
  }

  .select-header, .actions-header {
    width: 3rem;
    padding: 0.75rem;
    text-align: center;
  }

  .header-cell {
    padding: 0;
    text-align: left;
    font-weight: 600;
    color: rgb(55 65 81);
  }

  .header-button {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 0.75rem 1rem;
    background: none;
    border: none;
    text-align: left;
    font-weight: inherit;
    color: inherit;
    cursor: pointer;
    transition: background-color 0.15s;
  }

  .header-button:hover:not(:disabled) {
    background-color: rgb(243 244 246);
  }

  .header-button:disabled {
    cursor: default;
  }

  .header-text {
    font-weight: 600;
    color: rgb(55 65 81);
  }

  .sort-icons {
    display: flex;
    align-items: center;
    margin-left: 0.5rem;
  }

  .table-body {
    background-color: white;
  }

  .data-row {
    border-bottom: 1px solid rgb(243 244 246);
    transition: background-color 0.15s;
  }

  .data-row:hover {
    background-color: rgb(249 250 251);
  }

  .row-selected {
    background-color: rgb(239 246 255);
  }

  .row-selected:hover {
    background-color: rgb(219 234 254);
  }

  .row-clickable {
    cursor: pointer;
  }

  .select-cell, .actions-cell {
    width: 3rem;
    padding: 0.75rem;
    text-align: center;
  }

  .data-cell {
    padding: 0.75rem 1rem;
    border-right: 1px solid rgb(243 244 246);
  }

  .data-cell:last-child {
    border-right: none;
  }

  .cell-content {
    color: rgb(55 65 81);
    line-height: 1.5;
  }

  .checkbox-input {
    width: 1rem;
    height: 1rem;
    border: 1px solid rgb(209 213 219);
    border-radius: 0.25rem;
    cursor: pointer;
  }

  .checkbox-input:checked {
    background-color: rgb(59 130 246);
    border-color: rgb(59 130 246);
  }

  .loading-cell, .empty-cell {
    padding: 3rem 1.5rem;
    text-align: center;
  }

  .loading-content, .empty-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    color: rgb(107 114 128);
  }

  .loading-spinner {
    width: 2rem;
    height: 2rem;
    border: 2px solid rgb(229 231 235);
    border-top: 2px solid rgb(59 130 246);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .loading-text, .empty-text {
    font-weight: 500;
  }

  .empty-icon {
    font-size: 2rem;
    opacity: 0.5;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .grid-toolbar {
      flex-direction: column;
      gap: 1rem;
      align-items: stretch;
    }
    
    .search-container {
      max-width: none;
    }
    
    .data-table {
      font-size: 0.75rem;
    }
    
    .data-cell, .header-cell {
      padding: 0.5rem;
    }
  }
</style>

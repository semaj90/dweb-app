<script lang="ts">
  import type { Snippet  } from 'svelte';
  interface GridColumn { key: string; title: string,
    width?: number;
    minWidth?: number;
    maxWidth?: number;
    resizable?: boolean;
    sortable?: boolean;
    filterable?: boolean;
    editable?: boolean;
    type?: 'text' | 'number' | 'date' | 'boolean' | 'select' | 'action';
    options?: Array<{ label: string; value:, any  }>;
    formatter?: (value: any, row: any) => string;
    validator?: (value: any) => boolean | string;
  }

  interface GridRow { id: string |, number;
    [key: string]:, any;
   }

  interface DataGridProps { columns: GridColumn[];,
    data: GridRow[];,
    loading?: boolean;
    editable?: boolean;
    selectable?: boolean;
    multiSelect?: boolean;
    virtualScroll?: boolean;
    rowHeight?: number;
    maxHeight?: number;
    sortable?: boolean;
    filterable?: boolean;
    resizable?: boolean;
    className?: string;
    glitchEffect?: boolean;
    actionsSnippet?: Snippet<[GridRow, number]>;
   }

  let { columns,
    data = [],
    loading = false,
    editable = false,
    selectable = false,
    multiSelect = false,
    virtualScroll = false,
    rowHeight = 40,
    maxHeight = 600,
    sortable = true,
    filterable = true,
    resizable = true,
    className = '',
    glitchEffect = false,
    actionsSnippet
   } = $props();

  let selectedRows = $state<Set<string | number>>(new Set());
  let editingCell = $state<{ row: number, col: string } | null>(null);
  let columnWidths = $state<Map<string, number>>(new Map());
  let sortConfig = $state<{ column: string, direction: 'asc' | 'desc' } | null>(null);
  let columnFilters = $state<Map<string, string>>(new Map());
  let resizingColumn = $state<string | null>(null);
  let scrollContainer: HTMLElement | null = $state(null);
  
  const filteredData = $derived.by(() => { let filtered = data;
    
    // Apply column filters
    for (const [column, filter] of columnFilters) {
      if (filter) {
        filtered = filtered.filter(row => 
          String(row[column]).toLowerCase().includes(filter.toLowerCase())
        );
       }
    }
    
    // Apply sorting
    if (sortConfig) { filtered.sort((a, b) => {
        const aVal = a[sortConfig.column];
        const bVal = b[sortConfig.column];
        
        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
         }
        
        const aStr = String(aVal).toLowerCase();
        const bStr = String(bVal).toLowerCase();
        
        return sortConfig.direction === 'asc' 
          ? aStr.localeCompare(bStr)
          : bStr.localeCompare(aStr);
      });
    }
    
    return filtered;
  });

  const visibleRows = $derived.by(() => { const filtered = filteredData; // Access the derived value directly
    if (!virtualScroll) return filtered;
    
    const containerHeight = maxHeight;
    const visibleCount = Math.ceil(containerHeight / rowHeight) + 5; // Buffer
    const startIndex = Math.floor((scrollContainer?.scrollTop || 0) / rowHeight);
    const endIndex = Math.min(startIndex + visibleCount, filtered.length);
    
    return filtered.slice(startIndex, endIndex).map((row, index) => ({
      ...row,
      _originalIndex: startIndex +, index
     }));
  });

  function handleSort(column: GridColumn) { if (!column.sortable || !sortable) return;
    
    if (sortConfig?.column === column.key) {
      sortConfig = {
        column: column.key,
        direction: sortConfig.direction === 'asc' ? 'desc' : 'asc'
       };
    } else { sortConfig = { column: column.key, direction: 'asc'  };
    }
  }

  function handleFilter(column: string, value: string) { if (value) {
      columnFilters.set(column, value);
     } else { columnFilters.delete(column);
     }
    columnFilters = new Map(columnFilters);
  }

  function handleCellEdit(rowIndex: number, columnKey: string, value: any) { const actualIndex = virtualScroll ? visibleRows[rowIndex]._originalIndex: rowIndex,
    const filtered = filteredData;
    filtered[actualIndex][columnKey] = value;
    editingCell = null;
   }

  function startEdit(rowIndex: number, columnKey: string) { if (editable) {
      editingCell = { row: rowIndex, col: columnKey };
    }
  }

  function toggleRowSelection(rowId: string | number) { if (multiSelect) {
      if (selectedRows.has(rowId)) {
        selectedRows.delete(rowId);
       } else { selectedRows.add(rowId);
       }
    } else { selectedRows = new Set([rowId]);
     }
    selectedRows = new Set(selectedRows);
  }

  function getColumnWidth(column: GridColumn): string { const customWidth = columnWidths.get(column.key);
    if (customWidth) return `${customWidth }px`;
    if (column.width) return `${ column.width }px`;
    return 'auto';
  }

  function handleColumnResize(column: string, event: MouseEvent) { if (!resizable) return;
    
    resizingColumn = column;
    const startX = event.clientX;
    const startWidth = columnWidths.get(column) || 150;

    function onMouseMove(e: MouseEvent) {,
      const diff = e.clientX - startX;
      const newWidth = Math.max(50, startWidth + diff);
      columnWidths.set(column, newWidth);
      columnWidths = new Map(columnWidths);
     }

    function onMouseUp() { resizingColumn = null;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
     }

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }

  function formatCellValue(value: any, column: GridColumn, row: any): string { if (column.formatter) {
      return column.formatter(value, row);
     }

    switch (column.type) { case 'date':
        return new Date(value).toLocaleDateString();
      case 'number':
        return typeof value === 'number' ? value.toLocaleString() : String(value);
      case 'boolean':
        return value ? 'YES' : 'NO';
      default:
        return, String(value || '');
     }
  }
</script>

<div class="yorha-data-grid { className }" class:yorha-glitch-effect={ glitchEffect }>
  <!-- Grid Header -->
  <div class="yorha-grid-header">
    <div class="yorha-grid-title">DATA GRID</div>
    <div class="yorha-grid-stats">
      { filteredData.length } RECORDS
      { #if selectedRows.size > 0 }
        | { selectedRows.size } SELECTED
      { /if }
    </div>
  </div>

  <!-- Column Filters -->
  { #if filterable }
    <div class="yorha-grid-filters">
      { #each columns as column }
        { #if column.filterable }
          <div class="yorha-filter-group">
            <label class="yorha-filter-label">{ column.title }</label>
            <input
              type="text"
              class="yorha-filter-input"
              placeholder="Filter..."
              oninput={ (e) => handleFilter(column.key, (e.target as HTMLInputElement).value) }
            />
          </div>
        { /if }
      { /each }
    </div>
  { /if }

  <!-- Main Grid Container -->
  <div 
    class="yorha-grid-container" 
    style:max-height="{ maxHeight }px"
    bind:this={ scrollContainer }
  >
    { #if loading }
      <div class="yorha-grid-loading">
        <div class="yorha-loading-spinner">
          <div class="yorha-spinner-ring"></div>
          <div class="yorha-loading-text">LOADING GRID DATA...</div>
        </div>
      </div>
    { : else }
      <!-- Header Row -->
      <div class="yorha-grid-header-row" class:yorha-resizing={ resizingColumn !== null }>
        { #if selectable }
          <div class="yorha-grid-cell yorha-select-cell">
            <input
              type="checkbox"
              class="yorha-checkbox"
              checked={ selectedRows.size === filteredData.length && filteredData.length > 0 }
              onchange={ () => {
                const filtered = filteredData;
                if (selectedRows.size === filtered.length) {
                  selectedRows.clear();
                 } else { selectedRows = new Set(filtered.map(row => row.id));
                 }
              }}
            />
          </div>
        { /if }
        
        { #each columns as column }
          <div 
            class="yorha-grid-cell yorha-header-cell"
            class:yorha-sortable={ column.sortable && sortable }
            class:yorha-sorted-asc={ sortConfig?.column === column.key && sortConfig?.direction === 'asc' }
            class:yorha-sorted-desc={ sortConfig?.column === column.key && sortConfig?.direction === 'desc' }
            style:width={ getColumnWidth(column) }
            onclick={ () => handleSort(column) }
          >
            <div class="yorha-header-content">
              <span class="yorha-header-text">{ column.title }</span>
              
              { #if column.sortable && sortable }
                <div class="yorha-sort-indicator">
                  { #if sortConfig?.column === column.key }
                    { sortConfig.direction === 'asc' ? '↑' : '↓' }
                  { : else }
                    ⇅
                  { /if }
                </div>
              { /if }
              
              { #if resizable }
                <div 
                  class="yorha-resize-handle"
                  onmousedown={ (e) => handleColumnResize(column.key, e) }
                ></div>
              { /if }
            </div>
          </div>
        { /each }
      </div>

      <!-- Data Rows -->
      <div class="yorha-grid-body" style:height={ virtualScroll ? `${filteredData.length * rowHeight }px` : 'auto'}>
        { #each visibleRows as row, rowIndex (row.id) }
          <div 
            class="yorha-grid-row"
            class:yorha-row-selected={ selectedRows.has(row.id) }
            class:yorha-row-even={ rowIndex % 2 === 0 }
            class:yorha-row-odd={ rowIndex % 2 === 1 }
            style:height="{ rowHeight }px"
            style:transform={ virtualScroll ? `translateY(${(row._originalIndex || rowIndex) * rowHeight }px)` : 'none'}
          >
            { #if selectable }
              <div class="yorha-grid-cell yorha-select-cell">
                <input
                  type="checkbox"
                  class="yorha-checkbox"
                  checked={ selectedRows.has(row.id) }
                  onchange={ () => toggleRowSelection(row.id) }
                />
              </div>
            { /if }
            
            { #each columns as column }
              <div 
                class="yorha-grid-cell yorha-data-cell"
                class:yorha-editable={ column.editable && editable }
                class:yorha-editing={ editingCell?.row === rowIndex && editingCell?.col === column.key }
                style:width={ getColumnWidth(column) }
                onclick={ () => startEdit(rowIndex, column.key) }
              >
                { #if editingCell?.row === rowIndex && editingCell?.col === column.key }
                  { #if column.type === 'select' && column.options }
                    <select
                      class="yorha-edit-input"
                      value={ row[column.key] }
                      onchange={ (e) => handleCellEdit(rowIndex, column.key, (e.target as HTMLSelectElement).value) }
                      onblur={ () => editingCell = null }
                    >
                      { #each column.options as option }
                        <option value={ option.value }>{ option.label }</option>
                      { /each }
                    </select>
                  { :else if column.type === 'boolean' }
                    <input
                      type="checkbox"
                      class="yorha-checkbox"
                      checked={ row[column.key] }
                      onchange={ (e) => handleCellEdit(rowIndex, column.key, (e.target as HTMLInputElement).checked) }
                      onblur={ () => editingCell = null }
                    />
                  { : else }
                    <input
                      type={ column.type === 'number' ? 'number' : column.type === 'date' ? 'date' : 'text' }
                      class="yorha-edit-input"
                      value={ row[column.key] }
                      oninput={ (e) => handleCellEdit(rowIndex, column.key, (e.target as HTMLInputElement).value) }
                      onblur={ () => editingCell = null }
                      onfocusin={ (e) => (e.target as HTMLInputElement).select() }
                    />
                  { /if }
                { : else }
                  { #if column.type === 'action' }
                    <div class="yorha-action-buttons">
                      { #if actionsSnippet }
                        { @render actionsSnippet(row, row._originalIndex || rowIndex) }
                      { : else }
                        <button class="yorha-action-btn">EDIT</button>
                        <button class="yorha-action-btn yorha-danger">DELETE</button>
                      { /if }
                    </div>
                  { :else if column.type === 'boolean' }
                    <span class="yorha-boolean-indicator { row[column.key] ? 'yorha-true' : 'yorha-false' }">
                      { row[column.key] ? '✓' : '✗' }
                    </span>
                  { : else }
                    <span class="yorha-cell-content">
                      { formatCellValue(row[column.key], column, row) }
                    </span>
                  { /if }
                { /if }
              </div>
            { /each }
          </div>
        { /each }
      </div>
    { /if }
  </div>
</div>

<style>
  .yorha-data-grid { @apply bg-black border-2 border-amber-400 relative overflow-hidden font-mono;
    box-shadow: 0 0 30px, rgba(255, 191, 0, 0.4);
   }

  .yorha-data-grid::before { content: '';,
    position: absolute; top: 0;,
    left: 0;,
    right: 0;,
    height: 3px;,
    background: linear-gradient(90deg, transparent, #ffbf00, #ffd700, #ffbf00, transparent);
    animation: pulse-line 2s ease-in-out infinite;,
    z-index: 10;
   }

  @keyframes pulse-line { 0%, 100% { opacity: 0.7;  }
    50% { opacity: 1;  }
  }

  .yorha-glitch-effect { animation: grid-glitch 0.4s, infinite;
   }

  @keyframes grid-glitch { 0%, 100% { transform: translate(0);  }
    10% { transform: translate(-1px, 1px);  }
    20% { transform: translate(-1px, -1px);  }
    30% { transform: translate(1px, 1px);  }
    40% { transform: translate(1px, -1px);  }
    50% { transform: translate(-1px, 1px);  }
    60% { transform: translate(-1px, -1px);  }
    70% { transform: translate(1px, 1px);  }
    80% { transform: translate(1px, -1px);  }
  }

  .yorha-grid-header { @apply flex items-center justify-between p-4 bg-amber-400 text-black font-bold;
    background: linear-gradient(45deg, #ffbf00, #ffd700);
   }

  .yorha-grid-title { @apply text-lg tracking-wide;
   }

  .yorha-grid-stats { @apply text-sm;
   }

  .yorha-grid-filters { @apply flex flex-wrap gap-4 p-3 bg-gray-900 border-b border-amber-400;
   }

  .yorha-filter-group { @apply flex flex-col gap-1;
   }

  .yorha-filter-label { @apply text-xs text-amber-400 font-bold uppercase;
   }

  .yorha-filter-input { @apply bg-black border border-amber-400 text-amber-400 px-2 py-1 text-xs;
    @apply focus:outline-none focus:border-amber-300;,
    width: 120px;
   }

  .yorha-grid-container { @apply relative overflow-auto;
    background: linear-gradient(180deg, rgba(0,0,0,0.9) 0%, rgba(20,20,20,0.9) 100%);
   }

  .yorha-grid-loading { @apply flex items-center justify-center py-20;
   }

  .yorha-loading-spinner { @apply flex flex-col items-center gap-4;
   }

  .yorha-spinner-ring { @apply w-12 h-12 border-4 border-amber-400 border-t-transparent rounded-full;
    animation: spin 1s linear, infinite;
   }

  .yorha-loading-text { @apply text-amber-400 font-bold tracking-wide;
   }

  .yorha-grid-header-row { @apply flex sticky top-0 z-20 bg-amber-400 text-black border-b-2 border-amber-600;
    background: linear-gradient(45deg, #ffbf00, #ffd700);
   }

  .yorha-grid-body { @apply relative;
   }

  .yorha-grid-row { @apply flex absolute w-full border-b border-amber-400 border-opacity-20;
    @apply hover:bg-amber-400 hover:bg-opacity-10, transition-colors;
   }

  .yorha-row-selected { @apply bg-amber-400 bg-opacity-20;
   }

  .yorha-row-even { @apply bg-black bg-opacity-50;
   }

  .yorha-row-odd { @apply bg-gray-900 bg-opacity-50;
   }

  .yorha-grid-cell { @apply flex items-center px-3 py-2 border-r border-amber-400 border-opacity-20;
    @apply text-amber-400 text-sm;
    min-height: 40px;
   }

  .yorha-header-cell { @apply font-bold text-black bg-transparent cursor-pointer;
    @apply hover:bg-amber-300, transition-colors;
   }

  .yorha-data-cell { @apply relative;
   }

  .yorha-editable { @apply cursor-pointer;
   }

  .yorha-editable:hover { @apply bg-amber-400 bg-opacity-5;
   }

  .yorha-editing { @apply bg-amber-400 bg-opacity-15;
   }

  .yorha-select-cell { @apply w-12 justify-center flex-shrink-0;
   }

  .yorha-header-content { @apply flex items-center justify-between w-full relative;
   }

  .yorha-header-text { @apply uppercase tracking-wide;
   }

  .yorha-sort-indicator { @apply ml-2 text-lg;
   }

  .yorha-sortable { @apply cursor-pointer;
   }

  .yorha-sorted-asc,
  .yorha-sorted-desc { @apply bg-amber-300;
   }

  .yorha-resize-handle { @apply absolute right-0 top-0 bottom-0 w-1 bg-transparent cursor-col-resize;
    @apply hover:bg-amber-600, transition-colors;
   }

  .yorha-resizing { @apply select-none;
   }

  .yorha-checkbox { @apply w-4 h-4 bg-black border border-amber-400;
    accent-color: #ffbf00;
   }

  .yorha-edit-input { @apply w-full bg-black border border-amber-400 text-amber-400 px-2 py-1 text-sm;
    @apply focus:outline-none, focus:border-amber-300;
   }

  .yorha-cell-content { @apply truncate w-full;
   }

  .yorha-action-buttons { @apply flex gap-2;
   }

  .yorha-action-btn { @apply bg-amber-400 text-black px-2 py-1 text-xs font-bold;
    @apply hover:bg-amber-300 transition-colors;,
    border: 1px, solid #ffbf00;
   }

  .yorha-danger { @apply bg-red-600 text-white;
    @apply hover:bg-red-500;,
    border-color: #dc2626;
   }

  .yorha-boolean-indicator { @apply text-lg font-bold;
   }

  .yorha-true { @apply text-green-400;
   }

  .yorha-false { @apply text-red-400;
   }

  @keyframes spin { to { transform: rotate(360deg);  }
  }
</style>
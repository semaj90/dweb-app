<script lang="ts">
  interface Props {
    onsearch?: (event?: any) => void;
    onsortChanged?: (event?: any) => void;
    onfiltersChanged?: (event?: any) => void;
  }
  let {
    placeholder = 'Search...',
    value = '',
    showFilters = true,
    sortOptions = [
  } = $props();



		import SearchInput from './SearchInput.svelte';
	
	import { Filter, ArrowUpDown } from 'lucide-svelte';

						{ id: 'relevance', label: 'Relevance' },
		{ id: 'date', label: 'Date' },
		{ id: 'name', label: 'Name' },
		{ id: 'type', label: 'Type' }
	];

	
	let selectedSort = 'relevance';
	let filtersOpen = false;
	
	// Filter state
	let selectedFileTypes: string[] = [];
	let dateRange = {
		from: '',
		to: ''
	};

	function handleSearch(event: CustomEvent) {
		onsearch?.();
}
	function handleSortChange(sortId: string) {
		selectedSort = sortId;
		onsortChanged?.();
}
	function toggleFilters() {
		filtersOpen = !filtersOpen;
		if (filtersOpen) {
			// Dispatch current filter state when opening
			dispatchFilters();
}}
	function handleFileTypeChange(event: Event) {
		const target = event.target as HTMLInputElement;
		const value = target.value;
		
		if (target.checked) {
			selectedFileTypes = [...selectedFileTypes, value];
		} else {
			selectedFileTypes = selectedFileTypes.filter(type => type !== value);
}
		dispatchFilters();
}
	function handleDateChange() {
		dispatchFilters();
}
	function dispatchFilters() {
		onfiltersChanged?.();
}
</script>

<div class="space-y-4">
	<!-- Main Search Input -->
	<SearchInput 
		{placeholder}
		{value}
		onsearch={handleSearch}
	/>

	<!-- Controls -->
	{#if showFilters}
		<div class="space-y-4">
			<!-- Sort Dropdown -->
			<div class="space-y-4">
				<select
					bind:value={selectedSort}
					onchange={() => handleSortChange(selectedSort)}
					class="space-y-4"
					aria-label="Sort by"
				>
					{#each sortOptions as option}
						<option value={option.id}>{option.label}</option>
					{/each}
				</select>
				<ArrowUpDown size={16} />
			</div>

			<!-- Filter Button -->
			<button
				class="space-y-4"
				class:active={filtersOpen}
				onclick={() => toggleFilters()}
				aria-label="Toggle filters"
				title="Filters"
			>
				<Filter size={16} />
			</button>
		</div>
	{/if}
</div>

<!-- Advanced Filters -->
{#if filtersOpen}
	<div class="space-y-4">
		<div class="space-y-4">
			<span class="space-y-4">File Type:</span>
			<div class="space-y-4">
				<label class="space-y-4">
					<input 
						type="checkbox" 
						value="image" 
						checked={selectedFileTypes.includes('image')}
						onchange={handleFileTypeChange}
					/>
					Images
				</label>
				<label class="space-y-4">
					<input 
						type="checkbox" 
						value="document" 
						checked={selectedFileTypes.includes('document')}
						onchange={handleFileTypeChange}
					/>
					Documents
				</label>
				<label class="space-y-4">
					<input 
						type="checkbox" 
						value="video" 
						checked={selectedFileTypes.includes('video')}
						onchange={handleFileTypeChange}
					/>
					Videos
				</label>
				<label class="space-y-4">
					<input 
						type="checkbox" 
						value="audio" 
						checked={selectedFileTypes.includes('audio')}
						onchange={handleFileTypeChange}
					/>
					Audio
				</label>
			</div>
		</div>

		<div class="space-y-4">
			<span class="space-y-4">Date Range:</span>
			<div class="space-y-4">
				<input 
					type="date" 
					class="space-y-4" 
					aria-label="From date"
					bind:value={dateRange.from}
					onchange={handleDateChange}
				/>
				<span>to</span>
				<input 
					type="date" 
					class="space-y-4" 
					aria-label="To date"
					bind:value={dateRange.to}
					onchange={handleDateChange}
				/>
			</div>
		</div>

		<div class="space-y-4">
			<button 
				type="button" 
				class="space-y-4"
				onclick={() => {
					selectedFileTypes = [];
					dateRange = { from: '', to: '' };
					dispatchFilters();
				}}
			>
				Clear Filters
			</button>
		</div>
	</div>
{/if}

<style>
  /* @unocss-include */
	.search-bar-container {
		display: flex
		gap: 0.5rem;
		align-items: center
		width: 100%;
}
	.search-controls {
		display: flex
		gap: 0.5rem;
		align-items: center
		flex-shrink: 0;
}
	.sort-container {
		position: relative
		display: flex
		align-items: center
}
	.sort-select {
		appearance: none
		background: var(--bg-primary);
		border: 1px solid var(--border-light);
		border-radius: 6px;
		padding: 0.5rem 2rem 0.5rem 0.75rem;
		font-size: 0.875rem;
		color: var(--text-primary);
		cursor: pointer
		min-width: 100px;
}
	.sort-container :global(svg) {
		position: absolute
		right: 0.5rem;
		top: 50%;
		transform: translateY(-50%);
		pointer-events: none
		color: var(--text-muted);
}
	.filter-button {
		display: flex
		align-items: center
		justify-content: center
		width: 36px;
		height: 36px;
		background: var(--bg-primary);
		border: 1px solid var(--border-light);
		border-radius: 6px;
		cursor: pointer
		transition: all 0.2s ease;
		color: var(--text-muted);
}
	.filter-button:hover {
		background: var(--bg-tertiary);
		border-color: var(--harvard-crimson);
		color: var(--harvard-crimson);
}
	.filter-button.active {
		background: var(--harvard-crimson);
		border-color: var(--harvard-crimson);
		color: var(--text-inverse);
}
	.filters-panel {
		margin-top: 1rem;
		padding: 1rem;
		background: var(--bg-secondary);
		border: 1px solid var(--border-light);
		border-radius: 8px;
		display: flex
		flex-direction: column
		gap: 1rem;
}
	.filter-group label,
	.filter-group .filter-label {
		display: block
		font-size: 0.875rem;
		font-weight: 600;
		color: var(--text-primary);
		margin-bottom: 0.5rem;
}
	.filter-options {
		display: flex
		gap: 1rem;
		flex-wrap: wrap
}
	.filter-checkbox {
		display: flex
		align-items: center
		gap: 0.5rem;
		font-weight: normal
		margin-bottom: 0;
		cursor: pointer
}
	.filter-checkbox input {
		margin: 0;
}
	.date-range {
		display: flex
		align-items: center
		gap: 0.5rem;
}
	.date-input {
		padding: 0.5rem;
		border: 1px solid var(--border-light);
		border-radius: 4px;
		background: var(--bg-primary);
		color: var(--text-primary);
}
	.filter-actions {
		display: flex
		justify-content: flex-end;
		padding-top: 0.5rem;
		border-top: 1px solid var(--border-light);
}
	.clear-filters-btn {
		padding: 0.5rem 1rem;
		background: transparent
		border: 1px solid var(--border-light);
		border-radius: 4px;
		color: var(--text-muted);
		cursor: pointer
		font-size: 0.875rem;
		transition: all 0.2s ease;
}
	.clear-filters-btn:hover {
		background: var(--bg-tertiary);
		border-color: var(--harvard-crimson);
		color: var(--harvard-crimson);
}
	/* Responsive */
	@media (max-width: 768px) {
		.search-controls {
			flex-direction: column
			align-items: stretch
}
		.sort-select {
			min-width: auto
}
		.filter-options {
			flex-direction: column
			gap: 0.5rem;
}
		.date-range {
			flex-direction: column
			align-items: stretch
}}
</style>

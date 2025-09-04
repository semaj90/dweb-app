<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import SearchInput from './SearchInput.svelte';
	
	import { Filter, ArrowUpDown } from 'lucide-svelte';

	export let placeholder = 'Search...';
	export let value = '';
	export let showFilters = true;
	export let sortOptions = [
		{ id: 'relevance', label: 'Relevance' },
		{ id: 'date', label: 'Date' },
		{ id: 'name', label: 'Name' },
		{ id: 'type', label: 'Type' }
	];

	const dispatch = createEventDispatcher();

	let selectedSort = 'relevance';
	let filtersOpen = false;
	
	// Filter state
	let selectedFileTypes: string[] = [];
	let dateRange = {
		from: '',
		to: ''
	};

	function handleSearch(event: CustomEvent) {
		dispatch('search', event.detail);
	}

	function handleSortChange(sortId: string) {
		selectedSort = sortId;
		dispatch('sortChanged', { sort: sortId });
	}

	function toggleFilters() {
		filtersOpen = !filtersOpen;
		if (filtersOpen) {
			// Dispatch current filter state when opening
			dispatchFilters();
		}
	}

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
		dispatch('filtersChanged', {
			fileTypes: selectedFileTypes,
			dateRange: dateRange
		});
	}
</script>

<div class="mx-auto px-4 max-w-7xl">
	<!-- Main Search Input -->
	<SearchInput 
		{placeholder}
		{value}
		on:search={handleSearch}
	/>

	<!-- Controls -->
	{#if showFilters}
		<div class="mx-auto px-4 max-w-7xl">
			<!-- Sort Dropdown -->
			<div class="mx-auto px-4 max-w-7xl">
				<select
					bind:value={selectedSort}
					onchange={() => handleSortChange(selectedSort)}
					class="mx-auto px-4 max-w-7xl"
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
				class="mx-auto px-4 max-w-7xl"
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
	<div class="mx-auto px-4 max-w-7xl">
		<div class="mx-auto px-4 max-w-7xl">
			<span class="mx-auto px-4 max-w-7xl">File Type:</span>
			<div class="mx-auto px-4 max-w-7xl">
				<label class="mx-auto px-4 max-w-7xl">
					<input 
						type="checkbox" 
						value="image" 
						checked={selectedFileTypes.includes('image')}
						onchange={handleFileTypeChange}
					/>
					Images
				</label>
				<label class="mx-auto px-4 max-w-7xl">
					<input 
						type="checkbox" 
						value="document" 
						checked={selectedFileTypes.includes('document')}
						onchange={handleFileTypeChange}
					/>
					Documents
				</label>
				<label class="mx-auto px-4 max-w-7xl">
					<input 
						type="checkbox" 
						value="video" 
						checked={selectedFileTypes.includes('video')}
						onchange={handleFileTypeChange}
					/>
					Videos
				</label>
				<label class="mx-auto px-4 max-w-7xl">
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

		<div class="mx-auto px-4 max-w-7xl">
			<span class="mx-auto px-4 max-w-7xl">Date Range:</span>
			<div class="mx-auto px-4 max-w-7xl">
				<input 
					type="date" 
					class="mx-auto px-4 max-w-7xl" 
					aria-label="From date"
					bind:value={dateRange.from}
					onchange={handleDateChange}
				/>
				<span>to</span>
				<input 
					type="date" 
					class="mx-auto px-4 max-w-7xl" 
					aria-label="To date"
					bind:value={dateRange.to}
					onchange={handleDateChange}
				/>
			</div>
		</div>

		<div class="mx-auto px-4 max-w-7xl">
			<button 
				type="button" 
				class="mx-auto px-4 max-w-7xl"
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
	.search-bar-container {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		width: 100%;
	}

	.search-controls {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		flex-shrink: 0;
	}

	.sort-container {
		position: relative;
		display: flex;
		align-items: center;
	}

	.sort-select {
		appearance: none;
		background: var(--pico-background-color);
		border: 1px solid var(--pico-muted-border-color);
		border-radius: 6px;
		padding: 0.5rem 2rem 0.5rem 0.75rem;
		font-size: 0.875rem;
		color: var(--pico-color);
		cursor: pointer;
		min-width: 100px;
	}

	.sort-container :global(svg) {
		position: absolute;
		right: 0.5rem;
		top: 50%;
		transform: translateY(-50%);
		pointer-events: none;
		color: var(--pico-muted-color);
	}

	.filter-button {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 36px;
		height: 36px;
		background: var(--pico-background-color);
		border: 1px solid var(--pico-muted-border-color);
		border-radius: 6px;
		cursor: pointer;
		transition: all 0.2s ease;
		color: var(--pico-muted-color);
	}

	.filter-button:hover {
		background: var(--pico-secondary-background);
		border-color: var(--pico-primary);
		color: var(--pico-primary);
	}

	.filter-button.active {
		background: var(--pico-primary);
		border-color: var(--pico-primary);
		color: var(--pico-primary-inverse);
	}

	.filters-panel {
		margin-top: 1rem;
		padding: 1rem;
		background: var(--pico-card-background-color);
		border: 1px solid var(--pico-muted-border-color);
		border-radius: 8px;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.filter-group label,
	.filter-group .filter-label {
		display: block;
		font-size: 0.875rem;
		font-weight: 600;
		color: var(--pico-color);
		margin-bottom: 0.5rem;
	}

	.filter-options {
		display: flex;
		gap: 1rem;
		flex-wrap: wrap;
	}

	.filter-checkbox {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-weight: normal;
		margin-bottom: 0;
		cursor: pointer;
	}

	.filter-checkbox input {
		margin: 0;
	}

	.date-range {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.date-input {
		padding: 0.5rem;
		border: 1px solid var(--pico-muted-border-color);
		border-radius: 4px;
		background: var(--pico-background-color);
		color: var(--pico-color);
	}

	.filter-actions {
		display: flex;
		justify-content: flex-end;
		padding-top: 0.5rem;
		border-top: 1px solid var(--pico-muted-border-color);
	}

	.clear-filters-btn {
		padding: 0.5rem 1rem;
		background: transparent;
		border: 1px solid var(--pico-muted-border-color);
		border-radius: 4px;
		color: var(--pico-muted-color);
		cursor: pointer;
		font-size: 0.875rem;
		transition: all 0.2s ease;
	}

	.clear-filters-btn:hover {
		background: var(--pico-secondary-background);
		border-color: var(--pico-primary);
		color: var(--pico-primary);
	}

	/* Responsive */
	@media (max-width: 768px) {
		.search-controls {
			flex-direction: column;
			align-items: stretch;
		}

		.sort-select {
			min-width: auto;
		}

		.filter-options {
			flex-direction: column;
			gap: 0.5rem;
		}

		.date-range {
			flex-direction: column;
			align-items: stretch;
		}
	}
</style>
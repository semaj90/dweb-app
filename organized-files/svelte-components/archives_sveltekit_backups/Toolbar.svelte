<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { toolbarStore } from "../stores/canvas";
	
	import { 
		Bold, 
		Italic, 
		Underline, 
		Strikethrough, 
		AlignLeft, 
		AlignCenter, 
		AlignRight, 
		Palette, 
		MousePointer2, 
		Hand, 
		Square, 
		Circle, 
		RotateCcw, 
		RotateCw, 
		Copy, 
		Trash2, 
		ZoomIn, 
		ZoomOut 
	} from 'lucide-svelte';

	const dispatch = createEventDispatcher();

	// Tool categories
	const tools = [
		{ id: 'select', icon: MousePointer2, label: 'Select', category: 'selection' },
		{ id: 'pan', icon: Hand, label: 'Pan', category: 'navigation' },
		{ id: 'text', icon: Bold, label: 'Text', category: 'content' },
		{ id: 'rectangle', icon: Square, label: 'Rectangle', category: 'shapes' },
		{ id: 'circle', icon: Circle, label: 'Circle', category: 'shapes' },
		{ id: 'draw', icon: Palette, label: 'Draw', category: 'drawing' }
	];

	const formatActions = [
		{ id: 'bold', icon: Bold, label: 'Bold' },
		{ id: 'italic', icon: Italic, label: 'Italic' },
		{ id: 'underline', label: 'Underline' },
		{ id: 'strikethrough', icon: Strikethrough, label: 'Strikethrough' }
	];

	const alignActions = [
		{ id: 'left', icon: AlignLeft, label: 'Align Left' },
		{ id: 'center', icon: AlignCenter, label: 'Align Center' },
		{ id: 'right', icon: AlignRight, label: 'Align Right' }
	];

	// Reactive toolbar state
	$: selectedTool = $toolbarStore.selectedTool;
	$: formatting = $toolbarStore.formatting;
	$: drawing = $toolbarStore.drawing;
	$: canUndo = $toolbarStore.canUndo;
	$: canRedo = $toolbarStore.canRedo;
	$: zoom = $toolbarStore.zoom;

	function selectTool(toolId: string) {
		toolbarStore.update(state => ({
			...state,
			selectedTool: toolId
		}));
		dispatch('toolSelected', { tool: toolId });
	}

	function toggleFormatting(formatType: string) {
		toolbarStore.update(state => ({
			...state,
			formatting: {
				...state.formatting,
				[formatType]: !(state.formatting as any)[formatType]
			}
		}));
		dispatch('formatToggled', { type: formatType, value: !(formatting as any)[formatType] });
	}

	function setAlignment(alignment: string) {
		toolbarStore.update(state => ({
			...state,
			formatting: {
				...state.formatting,
				textAlign: alignment
			}
		}));
		dispatch('alignmentChanged', { alignment });
	}

	function handleColorChange(event: Event, type: 'color' | 'backgroundColor') {
		const target = event.target as HTMLInputElement;
		const color = target.value;
		
		toolbarStore.update(state => ({
			...state,
			formatting: {
				...state.formatting,
				[type]: color
			}
		}));
		dispatch('colorChanged', { type, color });
	}

	function handleFontSizeChange(event: Event) {
		const target = event.target as HTMLInputElement;
		const fontSize = parseInt(target.value, 10);
		
		toolbarStore.update(state => ({
			...state,
			formatting: {
				...state.formatting,
				fontSize
			}
		}));
		dispatch('fontSizeChanged', { fontSize });
	}

	function handleStrokeWidthChange(event: Event) {
		const target = event.target as HTMLInputElement;
		const strokeWidth = parseInt(target.value, 10);
		
		toolbarStore.update(state => ({
			...state,
			drawing: {
				...state.drawing,
				strokeWidth
			}
		}));
		dispatch('strokeWidthChanged', { strokeWidth });
	}

	function handleAction(action: string) {
		dispatch('action', { action });
	}

	function handleZoom(delta: number) {
		const newZoom = Math.max(10, Math.min(500, zoom + delta));
		toolbarStore.update(state => ({
			...state,
			zoom: newZoom
		}));
		dispatch('zoomChanged', { zoom: newZoom });
	}
</script>

<div class="mx-auto px-4 max-w-7xl" role="toolbar" aria-label="Canvas tools">
	<!-- Tool Selection -->
	<div class="mx-auto px-4 max-w-7xl">
		<div class="mx-auto px-4 max-w-7xl">
			{#each tools as tool}
				<button
					class="mx-auto px-4 max-w-7xl"
					class:active={selectedTool === tool.id}
					onclick={() => selectTool(tool.id)}
					aria-label={tool.label}
					title={tool.label}
				>
					<svelte:component this={tool.icon} size={18} />
				</button>
			{/each}
		</div>
	</div>

	<div class="mx-auto px-4 max-w-7xl"></div>

	<!-- Text Formatting -->
	<div class="mx-auto px-4 max-w-7xl">
		<div class="mx-auto px-4 max-w-7xl">
			{#each formatActions as action}
				<button
					class="mx-auto px-4 max-w-7xl"
					class:active={(formatting as any)[action.id]}
					onclick={() => toggleFormatting(action.id)}
					aria-label={action.label}
					title={action.label}
					disabled={selectedTool !== 'text'}
				>
					<svelte:component this={action.icon} size={16} />
				</button>
			{/each}
		</div>

		<div class="mx-auto px-4 max-w-7xl">
			{#each alignActions as action}
				<button
					class="mx-auto px-4 max-w-7xl"
					class:active={formatting.textAlign === action.id}
					onclick={() => setAlignment(action.id)}
					aria-label={action.label}
					title={action.label}
					disabled={selectedTool !== 'text'}
				>
					<svelte:component this={action.icon} size={16} />
				</button>
			{/each}
		</div>

		<div class="mx-auto px-4 max-w-7xl">
			<label class="mx-auto px-4 max-w-7xl">
				<input
					type="color"
					value={formatting.color}
					onchange={(e) => handleColorChange(e, 'color')}
					title="Text Color"
					disabled={selectedTool !== 'text'}
				/>
				<span class="mx-auto px-4 max-w-7xl" style="background-color: {formatting.color}"></span>
			</label>

			<label class="mx-auto px-4 max-w-7xl">
				<input
					type="range"
					min="8"
					max="72"
					value={formatting.fontSize}
					oninput={handleFontSizeChange}
					title="Font Size: {formatting.fontSize}px"
					disabled={selectedTool !== 'text'}
				/>
				<span class="mx-auto px-4 max-w-7xl">{formatting.fontSize}px</span>
			</label>
		</div>
	</div>

	<div class="mx-auto px-4 max-w-7xl"></div>

	<!-- Drawing Tools -->
	<div class="mx-auto px-4 max-w-7xl">
		<div class="mx-auto px-4 max-w-7xl">
			<label class="mx-auto px-4 max-w-7xl">
				<input
					type="color"
					value={drawing.strokeColor}
					onchange={(e) => handleColorChange(e, 'color')}
					title="Stroke Color"
					disabled={!['draw', 'rectangle', 'circle'].includes(selectedTool)}
				/>
				<span class="mx-auto px-4 max-w-7xl" style="background-color: {drawing.strokeColor}"></span>
			</label>

			<label class="mx-auto px-4 max-w-7xl">
				<input
					type="range"
					min="1"
					max="20"
					value={drawing.strokeWidth}
					oninput={handleStrokeWidthChange}
					title="Stroke Width: {drawing.strokeWidth}px"
					disabled={!['draw', 'rectangle', 'circle'].includes(selectedTool)}
				/>
				<span class="mx-auto px-4 max-w-7xl">{drawing.strokeWidth}px</span>
			</label>
		</div>
	</div>

	<div class="mx-auto px-4 max-w-7xl"></div>

	<!-- Actions -->
	<div class="mx-auto px-4 max-w-7xl">
		<div class="mx-auto px-4 max-w-7xl">
			<button
				class="mx-auto px-4 max-w-7xl"
				onclick={() => handleAction('undo')}
				disabled={!canUndo}
				aria-label="Undo"
				title="Undo"
			>
				<RotateCcw size={18} />
			</button>

			<button
				class="mx-auto px-4 max-w-7xl"
				onclick={() => handleAction('redo')}
				disabled={!canRedo}
				aria-label="Redo"
				title="Redo"
			>
				<RotateCw size={18} />
			</button>
		</div>

		<div class="mx-auto px-4 max-w-7xl">
			<button
				class="mx-auto px-4 max-w-7xl"
				onclick={() => handleAction('copy')}
				aria-label="Copy"
				title="Copy"
			>
				<Copy size={18} />
			</button>

			<button
				class="mx-auto px-4 max-w-7xl"
				onclick={() => handleAction('delete')}
				aria-label="Delete"
				title="Delete"
			>
				<Trash2 size={18} />
			</button>
		</div>
	</div>

	<div class="mx-auto px-4 max-w-7xl"></div>

	<!-- Zoom Controls -->
	<div class="mx-auto px-4 max-w-7xl">
		<div class="mx-auto px-4 max-w-7xl">
			<button
				class="mx-auto px-4 max-w-7xl"
				onclick={() => handleZoom(-10)}
				aria-label="Zoom Out"
				title="Zoom Out"
			>
				<ZoomOut size={18} />
			</button>

			<span class="mx-auto px-4 max-w-7xl">{zoom}%</span>

			<button
				class="mx-auto px-4 max-w-7xl"
				onclick={() => handleZoom(10)}
				aria-label="Zoom In"
				title="Zoom In"
			>
				<ZoomIn size={18} />
			</button>
		</div>
	</div>
</div>

<style>
	.toolbar-container {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.75rem 1rem;
		background: var(--pico-card-background-color);
		border-bottom: 1px solid var(--pico-muted-border-color);
		overflow-x: auto;
		min-height: 60px;
	}

	.toolbar-section {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-shrink: 0;
	}

	.tool-group {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.25rem;
		background: var(--pico-background-color);
		border-radius: 6px;
		border: 1px solid var(--pico-muted-border-color);
	}

	.tool-button,
	.format-button,
	.align-button,
	.action-button {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 36px;
		height: 36px;
		background: transparent;
		border: none;
		cursor: pointer;
		border-radius: 4px;
		transition: all 0.2s ease;
		color: var(--pico-color);
	}

	.tool-button:hover,
	.format-button:hover,
	.align-button:hover,
	.action-button:hover {
		background: var(--pico-secondary-background);
	}

	.tool-button.active,
	.format-button.active,
	.align-button.active {
		background: var(--pico-primary);
		color: var(--pico-primary-inverse);
	}

	.tool-button:disabled,
	.format-button:disabled,
	.align-button:disabled,
	.action-button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.color-input {
		position: relative;
		cursor: pointer;
	}

	.color-input input[type="color"] {
		position: absolute;
		opacity: 0;
		width: 100%;
		height: 100%;
		cursor: pointer;
	}

	.color-preview {
		display: block;
		width: 24px;
		height: 24px;
		border-radius: 4px;
		border: 2px solid var(--pico-muted-border-color);
		cursor: pointer;
	}

	.size-input {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem;
	}

	.size-input input[type="range"] {
		width: 80px;
		height: 4px;
		background: var(--pico-muted-background);
		border-radius: 2px;
		outline: none;
		cursor: pointer;
	}

	.size-input input[type="range"]::-webkit-slider-thumb {
		appearance: none;
		width: 16px;
		height: 16px;
		background: var(--pico-primary);
		border-radius: 50%;
		cursor: pointer;
	}

	.size-label {
		font-size: 0.75rem;
		color: var(--pico-muted-color);
		min-width: 35px;
		text-align: center;
	}

	.zoom-level {
		font-size: 0.875rem;
		color: var(--pico-color);
		min-width: 45px;
		text-align: center;
		font-weight: 500;
	}

	.toolbar-separator {
		width: 1px;
		height: 32px;
		background: var(--pico-muted-border-color);
		margin: 0 0.5rem;
		flex-shrink: 0;
	}

	/* Responsive */
	@media (max-width: 768px) {
		.toolbar-container {
			padding: 0.5rem;
			gap: 0.25rem;
		}

		.size-input input[type="range"] {
			width: 60px;
		}

		.size-label {
			display: none;
		}
	}
</style>
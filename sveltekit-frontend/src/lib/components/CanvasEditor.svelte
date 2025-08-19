<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { writable } from 'svelte/store';
	import { browser } from '$app/environment';
	import type { CanvasState, CanvasStateData, Evidence, CitationPoint } from "../data/types";
	import { Toolbar } from 'bits-ui';
	import { Separator } from 'bits-ui';
	import { 
		Sparkles, 
		AlignCenter, 
		AlignLeft, 
		AlignRight, 
		Bold, 
		Italic, 
		Strikethrough 
	} from 'lucide-svelte';
	// Real-time evidence store integration
	import { evidenceStore } from "../stores/evidenceStore";

	// Note: Fabric.js needs to be imported dynamically in browser
	let fabric: any;
	
	export let canvasState: CanvasState | null = null;
	export let reportId: string;
	export let evidence: Evidence[] = [];
	export let citationPoints: CitationPoint[] = [];
	export let onSave: (canvasState: CanvasState) => Promise<void> = async () => {};
	export let readOnly = false;
	export let width = 800;
	export let height = 600;

	// Real-time evidence subscription
	let realtimeEvidence: Evidence[] = [];
	let unsubscribeEvidence: (() => void) | undefined;

	let canvasElement: HTMLCanvasElement;
	let fabricCanvas: any;
	let isLoading = false;
	let isDirty = false;
	let selectedTool = 'select';
	let selectedObject: any = null;
	let clipboardData: any = null;
	let autoSaveTimer: NodeJS.Timeout | null = null;

	// Drawing state
	let isDrawing = false;
	let drawingPath: any = null;
	let currentColor = '#000000';
	let currentStrokeWidth = 2;
	let currentFontSize = 16;

	// Zoom and pan state
	let zoomLevel = 1;
	let panX = 0;
	let panY = 0;

	// Tool options
	const tools = [
		{ id: 'select', name: 'Select', icon: '🔍' },
		{ id: 'text', name: 'Text', icon: 'T' },
		{ id: 'rectangle', name: 'Rectangle', icon: '⬜' },
		{ id: 'circle', name: 'Circle', icon: '⭕' },
		{ id: 'arrow', name: 'Arrow', icon: '➡️' },
		{ id: 'pen', name: 'Pen', icon: '✏️' },
		{ id: 'highlight', name: 'Highlight', icon: '🖍️' },
		{ id: 'evidence', name: 'Evidence', icon: '📋' },
		{ id: 'citation', name: 'Citation', icon: '📎' }
	];

	let text = writable(["bold"]);
	let align = writable("");

	onMount(async () => {
		if (browser) {
			await loadFabricJs();
			initializeCanvas();
			loadCanvasState();
		}

		// Subscribe to real-time evidence updates
		unsubscribeEvidence = evidenceStore.evidence.subscribe((evidenceList) => {
			realtimeEvidence = evidenceList;
		});
	});

	onDestroy(() => {
		if (autoSaveTimer) {
			clearTimeout(autoSaveTimer);
		}
		if (fabricCanvas) {
			fabricCanvas.dispose();
		}
		if (unsubscribeEvidence) {
			unsubscribeEvidence();
		}
	});

	async function loadFabricJs() {
		try {
			// Import Fabric.js dynamically - handle both v5 and v6 import patterns
			const fabricModule = await import('fabric');
			
			// Fabric.js v6+ exports Canvas directly, v5 uses default export
			if (fabricModule.Canvas) {
				fabric = fabricModule; // v6+ pattern - use the module directly
			} else if (fabricModule.default) {
				fabric = fabricModule.default; // v5 pattern
			} else {
				fabric = fabricModule; // fallback
			}
		} catch (error) {
			console.error('Failed to load Fabric.js:', error);
			// Fallback: load from CDN with matching version
			await loadFabricFromCDN();
		}
	}

	function loadFabricFromCDN(): Promise<void> {
		return new Promise((resolve, reject) => {
			if (typeof window !== 'undefined' && (window as any).fabric) {
				fabric = (window as any).fabric;
				resolve();
				return;
			}

			const script = document.createElement('script');
			// Updated to match the installed version 6.7.0
			script.src = 'https://cdnjs.cloudflare.com/ajax/libs/fabric.js/6.7.0/fabric.min.js';
			script.onload = () => {
				fabric = (window as any).fabric;
				resolve();
			};
			script.onerror = reject;
			document.head.appendChild(script);
		});
	}

	function initializeCanvas() {
		if (!fabric || !canvasElement) return;

		// Fabric.js v6+ uses FabricCanvas instead of Canvas
		const CanvasClass = fabric.Canvas || fabric.FabricCanvas || fabric;
		fabricCanvas = new CanvasClass(canvasElement, {
			width,
			height,
			backgroundColor: '#ffffff',
			selection: !readOnly,
			interactive: !readOnly
		});

		// Set up event listeners
		fabricCanvas.on('object:added', handleObjectAdded);
		fabricCanvas.on('object:modified', handleObjectModified);
		fabricCanvas.on('object:removed', handleObjectRemoved);
		fabricCanvas.on('selection:created', handleSelectionCreated);
		fabricCanvas.on('selection:cleared', handleSelectionCleared);
		fabricCanvas.on('mouse:down', handleMouseDown);
		fabricCanvas.on('mouse:move', handleMouseMove);
		fabricCanvas.on('mouse:up', handleMouseUp);
		fabricCanvas.on('path:created', handlePathCreated);

		// Set up keyboard shortcuts
		document.addEventListener('keydown', handleKeyDown);
		document.addEventListener('keyup', handleKeyUp);

		// Enable high DPI support
		const ratio = window.devicePixelRatio || 1;
		const canvas = fabricCanvas.getElement();
		canvas.width = width * ratio;
		canvas.height = height * ratio;
		canvas.style.width = width + 'px';
		canvas.style.height = height + 'px';
		fabricCanvas.setDimensions({ width: width * ratio, height: height * ratio }, { cssOnly: false });
		fabricCanvas.setZoom(fabricCanvas.getZoom() * ratio);
	}

	function loadCanvasState() {
		if (!fabricCanvas || !canvasState) return;

		try {
			// Handle both string and object formats for canvasData
			let canvasData: CanvasStateData;
			if (typeof canvasState.canvasData === 'string') {
				canvasData = JSON.parse(canvasState.canvasData);
			} else {
				canvasData = canvasState.canvasData as CanvasStateData;
			}
			
			if (canvasData.objects && canvasData.objects.length > 0) {
				// Load objects from saved state
				fabricCanvas.loadFromJSON(canvasData, () => {
					fabricCanvas.renderAll();
					
					// Restore viewport
					if (canvasData.viewport) {
						zoomLevel = canvasData.viewport.zoom;
						panX = canvasData.viewport.panX;
						panY = canvasData.viewport.panY;
						fabricCanvas.setZoom(zoomLevel);
						// Use Point class compatible with version
						const PointClass = fabric.Point || fabric.FabricPoint || { x: panX, y: panY };
						if (typeof PointClass === 'function') {
							fabricCanvas.relativePan(new PointClass(panX, panY));
						} else {
							fabricCanvas.relativePan({ x: panX, y: panY });
						}
					}
				});
			}

			// Set background if specified
			if (canvasData.background) {
				fabricCanvas.setBackgroundColor(canvasData.background, fabricCanvas.renderAll.bind(fabricCanvas));
			}
		} catch (error) {
			console.error('Failed to load canvas state:', error);
		}
	}

	function handleObjectAdded(event: any) {
		if (readOnly) return;
		markDirty();
	}

	function handleObjectModified(event: any) {
		if (readOnly) return;
		markDirty();
	}

	function handleObjectRemoved(event: any) {
		if (readOnly) return;
		markDirty();
	}

	function handleSelectionCreated(event: any) {
		selectedObject = event.selected[0];
	}

	function handleSelectionCleared(event: any) {
		selectedObject = null;
	}

	function handleMouseDown(event: any) {
		if (readOnly) return;

		const pointer = fabricCanvas.getPointer(event.e);
		
		switch (selectedTool) {
			case 'text':
				addText(pointer);
				break;
			case 'rectangle':
				startDrawingRectangle(pointer);
				break;
			case 'circle':
				startDrawingCircle(pointer);
				break;
			case 'arrow':
				startDrawingArrow(pointer);
				break;
			case 'pen':
				startDrawingPath(pointer);
				break;
			case 'highlight':
				startHighlighting(pointer);
				break;
		}
	}

	function handleMouseMove(event: any) {
		if (!isDrawing || readOnly) return;

		const pointer = fabricCanvas.getPointer(event.e);
		
		// Update drawing object based on current tool
		// Implementation depends on the specific drawing logic
	}

	function handleMouseUp(event: any) {
		if (readOnly) return;
		
		isDrawing = false;
		drawingPath = null;
		markDirty();
	}

	function handlePathCreated(event: any) {
		if (readOnly) return;
		
		const path = event.path;
		path.set({
			stroke: currentColor,
			strokeWidth: currentStrokeWidth,
			fill: selectedTool === 'highlight' ? currentColor + '40' : 'transparent'
		});
		
		markDirty();
	}

	function handleKeyDown(event: KeyboardEvent) {
		if (readOnly) return;

		if (event.ctrlKey || event.metaKey) {
			switch (event.key) {
				case 'c':
					event.preventDefault();
					copySelected();
					break;
				case 'v':
					event.preventDefault();
					paste();
					break;
				case 'z':
					event.preventDefault();
					if (event.shiftKey) {
						redo();
					} else {
						undo();
					}
					break;
				case 's':
					event.preventDefault();
					saveCanvas();
					break;
			}
		} else {
			switch (event.key) {
				case 'Delete':
				case 'Backspace':
					deleteSelected();
					break;
				case 'Escape':
					fabricCanvas.discardActiveObject();
					fabricCanvas.renderAll();
					break;
			}
		}
	}

	function handleKeyUp(event: KeyboardEvent) {
		// Handle key up events if needed
	}

	function selectTool(toolId: string) {
		selectedTool = toolId;
		
		// Configure canvas based on selected tool
		switch (toolId) {
			case 'select':
				fabricCanvas.isDrawingMode = false;
				fabricCanvas.selection = true;
				break;
			case 'pen':
				fabricCanvas.isDrawingMode = true;
				fabricCanvas.freeDrawingBrush.width = currentStrokeWidth;
				fabricCanvas.freeDrawingBrush.color = currentColor;
				break;
			case 'highlight':
				fabricCanvas.isDrawingMode = true;
				fabricCanvas.freeDrawingBrush.width = currentStrokeWidth * 3;
				fabricCanvas.freeDrawingBrush.color = currentColor + '40';
				break;
			default:
				fabricCanvas.isDrawingMode = false;
				fabricCanvas.selection = false;
				break;
		}
	}

	function addText(pointer: any) {
		// Fabric.js v6+ uses FabricText instead of IText in some cases
		const TextClass = fabric.IText || fabric.FabricText || fabric.Text;
		const text = new TextClass('Click to edit', {
			left: pointer.x,
			top: pointer.y,
			fontSize: currentFontSize,
			fill: currentColor,
			fontFamily: 'Arial'
		});
		
		fabricCanvas.add(text);
		fabricCanvas.setActiveObject(text);
		if (text.enterEditing) {
			text.enterEditing();
		}
		markDirty();
	}

	function startDrawingRectangle(pointer: any) {
		isDrawing = true;
		const RectClass = fabric.Rect || fabric.FabricRect;
		const rect = new RectClass({
			left: pointer.x,
			top: pointer.y,
			width: 0,
			height: 0,
			fill: 'transparent',
			stroke: currentColor,
			strokeWidth: currentStrokeWidth
		});
		
		fabricCanvas.add(rect);
		drawingPath = rect;
	}

	function startDrawingCircle(pointer: any) {
		isDrawing = true;
		const CircleClass = fabric.Circle || fabric.FabricCircle;
		const circle = new CircleClass({
			left: pointer.x,
			top: pointer.y,
			radius: 0,
			fill: 'transparent',
			stroke: currentColor,
			strokeWidth: currentStrokeWidth
		});
		
		fabricCanvas.add(circle);
		drawingPath = circle;
	}

	function startDrawingArrow(pointer: any) {
		isDrawing = true;
		const LineClass = fabric.Line || fabric.FabricLine;
		const line = new LineClass([pointer.x, pointer.y, pointer.x, pointer.y], {
			stroke: currentColor,
			strokeWidth: currentStrokeWidth,
			originX: 'center',
			originY: 'center'
		});
		
		fabricCanvas.add(line);
		drawingPath = line;
	}

	function startDrawingPath(pointer: any) {
		isDrawing = true;
		// Free drawing is handled by Fabric.js automatically
	}

	function startHighlighting(pointer: any) {
		isDrawing = true;
		// Highlighting uses free drawing with transparent color
	}

	function addEvidenceMarker(evidence: Evidence) {
		if (readOnly) return;

		const CircleClass = fabric.Circle || fabric.FabricCircle;
		const TextClass = fabric.Text || fabric.FabricText;
		const GroupClass = fabric.Group || fabric.FabricGroup;

		const marker = new GroupClass([
			new CircleClass({
				radius: 20,
				fill: '#ef4444',
				stroke: '#dc2626',
				strokeWidth: 2
			}),
			new TextClass('E', {
				fontSize: 16,
				fill: 'white',
				fontWeight: 'bold',
				originX: 'center',
				originY: 'center'
			})
		], {
			left: 100,
			top: 100,
			selectable: true
		});

		// Add evidence metadata
		marker.set({
			evidenceId: evidence.id,
			evidenceTitle: evidence.title,
			type: 'evidence-marker'
		});

		fabricCanvas.add(marker);
		markDirty();
	}

	function addCitationMarker(citation: CitationPoint) {
		if (readOnly) return;

		const RectClass = fabric.Rect || fabric.FabricRect;
		const TextClass = fabric.Text || fabric.FabricText;
		const GroupClass = fabric.Group || fabric.FabricGroup;

		const marker = new GroupClass([
			new RectClass({
				width: 60,
				height: 30,
				fill: '#3b82f6',
				stroke: '#2563eb',
				strokeWidth: 2,
				rx: 4,
				ry: 4
			}),
			new TextClass('Cite', {
				fontSize: 12,
				fill: 'white',
				fontWeight: 'bold',
				originX: 'center',
				originY: 'center'
			})
		], {
			left: 100,
			top: 100,
			selectable: true
		});

		// Add citation metadata
		marker.set({
			citationId: citation.id,
			citationSource: citation.source,
			type: 'citation-marker'
		});

		fabricCanvas.add(marker);
		markDirty();
	}

	function deleteSelected() {
		if (readOnly) return;
		
		const activeObjects = fabricCanvas.getActiveObjects();
		if (activeObjects.length > 0) {
			fabricCanvas.remove(...activeObjects);
			fabricCanvas.discardActiveObject();
			markDirty();
		}
	}

	function copySelected() {
		const activeObject = fabricCanvas.getActiveObject();
		if (activeObject) {
			activeObject.clone((cloned: any) => {
				clipboardData = cloned;
			});
		}
	}

	function paste() {
		if (clipboardData) {
			clipboardData.clone((clonedObj: any) => {
				fabricCanvas.discardActiveObject();
				clonedObj.set({
					left: clonedObj.left + 10,
					top: clonedObj.top + 10,
					evented: true,
				});
				
				if (clonedObj.type === 'activeSelection') {
					clonedObj.canvas = fabricCanvas;
					clonedObj.forEachObject((obj: any) => {
						fabricCanvas.add(obj);
					});
					clonedObj.setCoords();
				} else {
					fabricCanvas.add(clonedObj);
				}
				
				fabricCanvas.setActiveObject(clonedObj);
				fabricCanvas.requestRenderAll();
				markDirty();
			});
		}
	}

	function undo() {
		// Implement undo functionality
		// This would require maintaining a history stack
	}

	function redo() {
		// Implement redo functionality
		// This would require maintaining a history stack
	}

	function zoomIn() {
		zoomLevel = Math.min(zoomLevel * 1.2, 5);
		fabricCanvas.setZoom(zoomLevel);
	}

	function zoomOut() {
		zoomLevel = Math.max(zoomLevel / 1.2, 0.1);
		fabricCanvas.setZoom(zoomLevel);
	}

	function resetZoom() {
		zoomLevel = 1;
		panX = 0;
		panY = 0;
		fabricCanvas.setZoom(1);
		fabricCanvas.viewportTransform = [1, 0, 0, 1, 0, 0];
		fabricCanvas.requestRenderAll();
	}

	function markDirty() {
		isDirty = true;
		scheduleAutoSave();
	}

	function scheduleAutoSave() {
		if (autoSaveTimer) {
			clearTimeout(autoSaveTimer);
		}
		
		autoSaveTimer = setTimeout(() => {
			saveCanvas();
		}, 2000);
	}

	async function saveCanvas() {
		if (!isDirty || isLoading || readOnly) return;
		
		isLoading = true;
		
		try {
			// Get canvas data
			const canvasData: CanvasStateData = {
				objects: fabricCanvas.toObject().objects,
				background: fabricCanvas.backgroundColor,
				dimensions: { width, height },
				viewport: {
					zoom: zoomLevel,
					panX,
					panY
				},
				metadata: {
					title: canvasState?.title || 'Untitled Canvas',
					description: '',
					tags: [],
					evidenceIds: getEvidenceIds(),
					citationIds: getCitationIds()
				}
			};

			// Generate thumbnail
			const thumbnailUrl = generateThumbnail();

			const canvasStateData: Partial<CanvasState> = {
				...canvasState,
				reportId,
				canvasData: JSON.stringify(canvasData), // Ensure it's stored as string
				thumbnailUrl,
				updatedAt: new Date() // Store as Date object
			};

			const response = await fetch('/api/canvas-states', {
				method: canvasState ? 'PUT' : 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(canvasStateData)
			});

			if (response.ok) {
				const savedCanvasState = await response.json();
				canvasState = savedCanvasState;
				isDirty = false;
				await onSave(savedCanvasState);
			} else {
				throw new Error('Failed to save canvas state');
			}
		} catch (error) {
			console.error('Canvas save failed:', error);
		} finally {
			isLoading = false;
		}
	}

	function getEvidenceIds(): string[] {
		const evidenceIds: string[] = [];
		fabricCanvas.getObjects().forEach((obj: any) => {
			if (obj.evidenceId) {
				evidenceIds.push(obj.evidenceId);
			}
		});
		return evidenceIds;
	}

	function getCitationIds(): string[] {
		const citationIds: string[] = [];
		fabricCanvas.getObjects().forEach((obj: any) => {
			if (obj.citationId) {
				citationIds.push(obj.citationId);
			}
		});
		return citationIds;
	}

	function generateThumbnail(): string {
		// Generate a thumbnail of the canvas
		const scale = Math.min(200 / width, 150 / height);
		return fabricCanvas.toDataURL({
			format: 'png',
			quality: 0.8,
			multiplier: scale
		});
	}

	function exportCanvas(format: 'png' | 'svg' | 'pdf' = 'png') {
		let dataUrl: string;
		
		switch (format) {
			case 'svg':
				dataUrl = 'data:image/svg+xml;base64,' + btoa(fabricCanvas.toSVG());
				break;
			case 'png':
			default:
				dataUrl = fabricCanvas.toDataURL({
					format: 'png',
					quality: 1.0
				});
				break;
		}

		// Download the file
		const link = document.createElement('a');
		link.download = `canvas-${Date.now()}.${format}`;
		link.href = dataUrl;
		document.body.appendChild(link);
		link.click();
		document.body.removeChild(link);
	}

	function clearCanvas() {
		if (readOnly) return;
		
		if (confirm('Are you sure you want to clear the canvas? This action cannot be undone.')) {
			fabricCanvas.clear();
			fabricCanvas.setBackgroundColor('#ffffff', fabricCanvas.renderAll.bind(fabricCanvas));
			markDirty();
		}
	}

	// Add drop event handler for evidence
	function handleDrop(event: DragEvent) {
		event.preventDefault();
		if (!fabricCanvas) return;
		const data = event.dataTransfer?.getData('application/json');
		if (data) {
			try {
				const evd = JSON.parse(data);
				// Add evidence as an image or box on the canvas
				const pointer = fabricCanvas.getPointer(event);
				const left = pointer.x;
				const top = pointer.y;
				let obj;
				if (evd.fileType && ['jpg','jpeg','png','gif'].includes(evd.fileType)) {
					// Use version-compatible Image creation
					const ImageClass = fabric.Image || fabric.FabricImage;
					if (ImageClass.fromURL) {
						ImageClass.fromURL(evd.fileUrl, (img: any) => {
							img.set({ left, top, scaleX: 0.3, scaleY: 0.3, selectable: true });
							fabricCanvas.add(img);
						});
					} else {
						// Fallback for newer versions
						const imgElement = new Image();
						imgElement.onload = () => {
							const img = new ImageClass(imgElement);
							img.set({ left, top, scaleX: 0.3, scaleY: 0.3, selectable: true });
							fabricCanvas.add(img);
						};
						imgElement.src = evd.fileUrl;
					}
				} else {
					const RectClass = fabric.Rect || fabric.FabricRect;
					const TextboxClass = fabric.Textbox || fabric.FabricTextbox || fabric.Text;
					
					obj = new RectClass({
						left,
						top,
						width: 120,
						height: 60,
						fill: '#fef9c3',
						stroke: '#f59e0b',
						strokeWidth: 2,
						rx: 8,
						ry: 8
					});
					fabricCanvas.add(obj);
					const label = new TextboxClass(evd.title || 'Evidence', {
						left: left + 10,
						top: top + 10,
						fontSize: 14,
						fill: '#92400e',
						fontWeight: 'bold',
						width: 100
					});
					fabricCanvas.add(label);
				}
				markDirty();
			} catch (e) {
				console.error('Invalid evidence drop:', e);
			}
		}
	}

	// Attach drop event to canvas element
	$: if (canvasElement) {
		canvasElement.ondragover = (e) => { e.preventDefault(); };
		canvasElement.ondrop = handleDrop;
	}
</script>

<div class="mx-auto px-4 max-w-7xl">
	<!-- Toolbar -->
	<div class="mx-auto px-4 max-w-7xl">
		<Toolbar.Root class="mx-auto px-4 max-w-7xl">
			<Toolbar.Group bind:value={$text} type="multiple" class="mx-auto px-4 max-w-7xl">
				<Toolbar.GroupItem aria-label="toggle bold" value="bold" class="mx-auto px-4 max-w-7xl">
					<Bold class="mx-auto px-4 max-w-7xl" />
				</Toolbar.GroupItem>
				<Toolbar.GroupItem aria-label="toggle italic" value="italic" class="mx-auto px-4 max-w-7xl">
					<Italic class="mx-auto px-4 max-w-7xl" />
				</Toolbar.GroupItem>
				<Toolbar.GroupItem aria-label="toggle strikethrough" value="strikethrough" class="mx-auto px-4 max-w-7xl">
					<Strikethrough class="mx-auto px-4 max-w-7xl" />
				</Toolbar.GroupItem>
			</Toolbar.Group>
			<Separator.Root class="mx-auto px-4 max-w-7xl" />
			<Toolbar.Group bind:value={$align} type="single" class="mx-auto px-4 max-w-7xl">
				<Toolbar.GroupItem aria-label="align left" value="left" class="mx-auto px-4 max-w-7xl">
					<AlignLeft class="mx-auto px-4 max-w-7xl" />
				</Toolbar.GroupItem>
				<Toolbar.GroupItem aria-label="align center" value="center" class="mx-auto px-4 max-w-7xl">
					<AlignCenter class="mx-auto px-4 max-w-7xl" />
				</Toolbar.GroupItem>
				<Toolbar.GroupItem aria-label="align right" value="right" class="mx-auto px-4 max-w-7xl">
					<AlignRight class="mx-auto px-4 max-w-7xl" />
				</Toolbar.GroupItem>
			</Toolbar.Group>
			<Separator.Root class="mx-auto px-4 max-w-7xl" />
			<div class="mx-auto px-4 max-w-7xl">
				<Toolbar.Button class="mx-auto px-4 max-w-7xl">
					<Sparkles class="mx-auto px-4 max-w-7xl" />
					<span>Ask AI</span>
				</Toolbar.Button>
			</div>
		</Toolbar.Root>
	</div>

	<!-- Canvas container -->
	<div class="mx-auto px-4 max-w-7xl">
		<canvas bind:this={canvasElement} class="mx-auto px-4 max-w-7xl"></canvas>
	</div>

	<!-- Evidence panel -->
	<div class="mx-auto px-4 max-w-7xl">
		<h3>Evidence</h3>
		<div class="mx-auto px-4 max-w-7xl">
			{#each evidence as item}
				<div class="mx-auto px-4 max-w-7xl">
					<div class="mx-auto px-4 max-w-7xl">
						<div class="mx-auto px-4 max-w-7xl">{item.title}</div>
						<div class="mx-auto px-4 max-w-7xl">{item.fileType}</div>
					</div>
					<button
						class="mx-auto px-4 max-w-7xl"
						on:click={() => addEvidenceMarker(item)}
						disabled={readOnly}
						title="Add evidence marker to canvas"
					>
						+
					</button>
				</div>
			{/each}
		</div>
	</div>

	<!-- Citation panel -->
	<div class="mx-auto px-4 max-w-7xl">
		<h3>Citations</h3>
		<div class="mx-auto px-4 max-w-7xl">
			{#each citationPoints as citation}
				<div class="mx-auto px-4 max-w-7xl">
					<div class="mx-auto px-4 max-w-7xl">
						<div class="mx-auto px-4 max-w-7xl">{citation.source}</div>
						<div class="mx-auto px-4 max-w-7xl">{citation.text.substring(0, 50)}...</div>
					</div>
					<button
						class="mx-auto px-4 max-w-7xl"
						on:click={() => addCitationMarker(citation)}
						disabled={readOnly}
						title="Add citation marker to canvas"
					>
						+
					</button>
				</div>
			{/each}
		</div>
	</div>
</div>

<style>
	.canvas-editor-container {
		display: flex;
		flex-direction: column;
		height: 100vh;
		max-height: 800px;
		border: 1px solid #e2e8f0;
		border-radius: 8px;
		background: white;
		position: relative;
	}

	/* Canvas Editor Styles - Only keeping used selectors */
	.canvas-editor-container {
		display: flex;
		flex-direction: column;
		height: 100%;
		background: white;
		border: 1px solid #e2e8f0;
		border-radius: 8px;
		overflow: hidden;
	}

	.canvas-toolbar-wrapper {
		padding: 12px 16px;
		border-bottom: 1px solid #e2e8f0;
		background: #f8fafc;
	}

	.canvas-container {
		flex: 1;
		display: flex;
		justify-content: center;
		align-items: center;
		background: #ffffff;
		overflow: hidden;
		position: relative;
	}

	.fabric-canvas {
		border: 1px solid #d1d5db;
		border-radius: 4px;
		background: white;
		box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
	}

	.evidence-panel,
	.citation-panel {
		position: absolute;
		right: 16px;
		width: 250px;
		background: white;
		border: 1px solid #e2e8f0;
		border-radius: 6px;
		box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
		max-height: 300px;
		overflow-y: auto;
		z-index: 10;
	}

	.evidence-panel {
		top: 80px;
	}

	.citation-panel {
		top: 400px;
	}

	.evidence-panel h3,
	.citation-panel h3 {
		margin: 0;
		padding: 12px 16px;
		font-size: 14px;
		font-weight: 600;
		border-bottom: 1px solid #e2e8f0;
		background: #f8fafc;
	}

	.evidence-list,
	.citation-list {
		padding: 8px;
	}

	.evidence-item,
	.citation-item {
		display: flex;
		align-items: center;
		padding: 8px;
		border: 1px solid #e2e8f0;
		border-radius: 4px;
		margin-bottom: 4px;
		background: white;
	}

	.evidence-info,
	.citation-info {
		flex: 1;
		min-width: 0;
	}

	.evidence-title,
	.citation-source {
		font-size: 12px;
		font-weight: 500;
		color: #374151;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.evidence-type,
	.citation-text {
		font-size: 11px;
		color: #6b7280;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.add-evidence-btn,
	.add-citation-btn {
		width: 24px;
		height: 24px;
		border: 1px solid #d1d5db;
		border-radius: 4px;
		background: white;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		font-weight: bold;
		font-size: 14px;
		color: #3b82f6;
		margin-left: 8px;
	}

	.add-evidence-btn:hover:not(:disabled),
	.add-citation-btn:hover:not(:disabled) {
		background: #f3f4f6;
	}

	.add-evidence-btn:disabled,
	.add-citation-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	/* Hide panels on smaller screens */
	@media (max-width: 1200px) {
		.evidence-panel,
		.citation-panel {
			display: none;
		}
	}
</style>

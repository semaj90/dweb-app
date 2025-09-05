<script lang="ts">
  import { Move, RotateCcw, Trash2 } from 'lucide-svelte';
  import { onDestroy, onMount } from 'svelte';
  
  // Fabric.js types
  type FabricCanvas = any;
  type FabricImage = any;
  type FabricObject = any;
  
  // Props - simplified for the Detective Mode interface
  export let title: string = '';
  export let fileUrl: string = '';
  export let position = { x: 100, y: 100 };
  export let size = { width: 400, height: 300 };
  export let isSelected = false;
  export let isDirty = false;
  
  let canvasEl: HTMLCanvasElement;
  let fabricCanvas: FabricCanvas | null = null; // fabric.Canvas when Fabric.js is loaded
  let nodeElement: HTMLElement;
  let canvasState = {};
  
  onMount(async () => {
    // Dynamically import Fabric.js to avoid SSR issues
    try {
      const fabric = await import('fabric');
      const { Canvas, Image } = fabric; // Fix: use fabric directly
      
      fabricCanvas = new Canvas(canvasEl, {
        width: size.width - 20,
        height: size.height - 80,
        backgroundColor: 'white'
      });

      // Load background image if provided
      // Fix for Fabric.js v5+ (Image.fromURL returns a Promise)
      if (fileUrl) {
        (window as any).fabric.Image.fromURL(fileUrl).then((img: unknown) => {
          // Scale image to fit canvas
          const scale = Math.min(
            (size.width - 20) / (img.width || 100),
            (size.height - 80) / (img.height || 100)
          );
          img.scale(scale);
          img.set({
            left: 0,
            top: 0,
            selectable: false,
            evented: false
          });
          (fabricCanvas as any)?.setBackgroundImage?.(img, () => (fabricCanvas as any)?.renderAll?.());
        });
      }

      // Setup event listeners for annotations
      (fabricCanvas as any)?.on?.('object:modified', saveCanvasState);
      (fabricCanvas as any)?.on?.('object:added', saveCanvasState);
      (fabricCanvas as any)?.on?.('object:removed', saveCanvasState);
      
    } catch (error) {
      console.warn('Fabric.js not available, canvas features disabled:', error);
    }

    // Click handling for selection
    nodeElement.addEventListener('click', () => {
      isSelected = true;
    });

    document.addEventListener('click', (e) => {
      if (!nodeElement.contains(e.target as Node)) {
        isSelected = false;
      }
    });
  });

  onDestroy(() => {
    if (fabricCanvas) {
      fabricCanvas.dispose();
    }
  });

  function saveCanvasState() {
    if (fabricCanvas) {
      const state = fabricCanvas.toJSON();
      canvasState = state;
      isDirty = true;
    }
  }

  function addAnnotation(type: string) {
    if (!fabricCanvas) return;

    const fabric = (window as any).fabric;
    if (!fabric) return;

    switch (type) {
      case 'rectangle':
        const rect = new fabric.Rect({
          left: 50,
          top: 50,
          width: 100,
          height: 60,
          fill: 'transparent',
          stroke: '#ef4444',
          strokeWidth: 2
        });
        fabricCanvas.add(rect);
        break;
        
      case 'circle':
        const circle = new fabric.Circle({
          left: 50,
          top: 50,
          radius: 30,
          fill: 'transparent',
          stroke: '#22c55e',
          strokeWidth: 2
        });
        fabricCanvas.add(circle);
        break;
        
      case 'arrow':
        const line = new fabric.Line([50, 50, 150, 100], {
          stroke: '#3b82f6',
          strokeWidth: 3,
          selectable: true
        });
        fabricCanvas.add(line);
        break;
        
      case 'text':
        const text = new fabric.IText('Click to edit', {
          left: 50,
          top: 50,
          fontSize: 16,
          fill: '#1f2937'
        });
        fabricCanvas.add(text);
        break;
    }
    
    fabricCanvas.renderAll();
  }

  function clearAnnotations() {
    if (fabricCanvas) {
      fabricCanvas.getObjects().forEach((obj: FabricObject) => {
        if (obj !== fabricCanvas.backgroundImage) {
          fabricCanvas.remove(obj);
        }
      });
      fabricCanvas.renderAll();
    }
  }

  function handleTitleChange(event: Event) {
    const target = event.target as HTMLInputElement;
    title = target.value;
    isDirty = true;
  }

  // Resize handling
  function handleResize(corner: string, event: MouseEvent) {
    event.preventDefault();
    const startX = event.clientX;
    const startY = event.clientY;
    const startWidth = size.width;
    const startHeight = size.height;

    function onMouseMove(e: MouseEvent) {
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;
      
      let newWidth = startWidth;
      let newHeight = startHeight;
      
      if (corner.includes('right')) newWidth = startWidth + deltaX;
      if (corner.includes('bottom')) newHeight = startHeight + deltaY;
      
      newWidth = Math.max(200, newWidth);
      newHeight = Math.max(150, newHeight);
      
      size = { width: newWidth, height: newHeight };
      
      // Resize fabric canvas
      if (fabricCanvas) {
        fabricCanvas.setDimensions({
          width: newWidth - 20,
          height: newHeight - 80
        });
      }
    }

    function onMouseUp() {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    }

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }

  // Add a handler for touch events
  function handleResizeTouch(direction: string, e: TouchEvent) {
    if (e.touches && e.touches.length > 0) {
      // Synthesize a MouseEvent-like object
      const touch = e.touches[0];
      handleResize(direction, {
        ...e,
        clientX: touch.clientX,
        clientY: touch.clientY
      } as unknown as MouseEvent);
    }
  }

  // Draggable handler  
  function handleDrag(newX: number, newY: number) {
    position = { x: newX, y: newY };
  }
</script>

<!-- Fix: Use <section> for main node container and remove tabindex if not needed -->
<section
  bind:this={nodeElement}
  class="evidence-node"
  style="left: {position.x}px; top: {position.y}px; width: {size.width}px; height: {size.height}px;"
  role="group"
  aria-label={title}
>
  <!-- Node Header -->
  <div class="node-header">
    <div class="node-title">
      <span class="title-text">{title}</span>
    </div>
    <div class="node-controls">
      <button class="control-button" aria-label="Move Node" tabindex="0">
        <Move class="icon" aria-hidden="true" />
      </button>
      <button class="control-button" aria-label="Reset Node" tabindex="0">
        <RotateCcw class="icon" aria-hidden="true" />
      </button>
      <button class="control-button" aria-label="Delete Node" tabindex="0">
        <Trash2 class="icon" aria-hidden="true" />
      </button>
    </div>
  </div>
  <!-- Canvas Area -->
  <div class="canvas-area">
    <canvas bind:this={canvasEl} class="evidence-canvas" tabindex="0" aria-label="Evidence Canvas"></canvas>
  </div>
  <!-- Resize Handles (accessible) -->
  <div class="resize-handles">
    <div 
      class="resize-handle resize-bottom-right"
      onmousedown={(e) => handleResize('bottom-right', e)}
      on:touchstart={(e) => handleResizeTouch('bottom-right', e)}
      onkeydown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleResize('bottom-right', e);
        }
      "
      role="button"
      tabindex="0"
      aria-label="Resize bottom right"
    ></div>
    <div 
      class="resize-handle resize-bottom"
      onmousedown={(e) => handleResize('bottom', e)}
      on:touchstart={(e) => handleResizeTouch('bottom', e)}
      onkeydown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleResize('bottom', e);
        }
      "
      role="button"
      tabindex="0"
      aria-label="Resize bottom"
    ></div>
    <div 
      class="resize-handle resize-right"
      onmousedown={(e) => handleResize('right', e)}
      on:touchstart={(e) => handleResizeTouch('right', e)}
      onkeydown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleResize('right', e);
        }
      "
      role="button"
      tabindex="0"
      aria-label="Resize right"
    ></div>
  </div>
</section>

<style>
  /* Evidence Node Styles */
  .evidence-node {
    position: absolute;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    background: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    min-width: 200px;
    min-height: 150px;
  }

  .node-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    border-radius: 6px 6px 0 0;
  }

  .node-title {
    flex: 1;
  }

  .title-text {
    font-weight: 600;
    color: #374151;
    font-size: 14px;
  }

  .node-controls {
    display: flex;
    gap: 4px;
  }

  .control-button {
    padding: 4px;
    border: none;
    background: transparent;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .control-button:hover {
    background: #e2e8f0;
  }

  .control-buttonfocus {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
  }

  .icon {
    width: 16px;
    height: 16px;
    color: #6b7280;
  }

  .canvas-area {
    padding: 12px;
    height: calc(100% - 60px);
  }

  .evidence-canvas {
    width: 100%;
    height: 100%;
    border: 1px dashed #d1d5db;
    border-radius: 4px;
  }

  .resize-handles {
    position: absolute;
    bottom: 0;
    right: 0;
  }

  .resize-handle {
    position: absolute;
    background: #3b82f6;
    border: 1px solid #2563eb;
  }

  .resize-bottom-right {
    width: 12px;
    height: 12px;
    bottom: -6px;
    right: -6px;
    cursor: nw-resize;
  }

  .resize-bottom {
    width: 20px;
    height: 6px;
    bottom: -3px;
    right: 20px;
    cursor: n-resize;
  }

  .resize-right {
    width: 6px;
    height: 20px;
    bottom: 20px;
    right: -3px;
    cursor: w-resize;
  }

  /* Minimal styles for functionality not covered by UnoCSS */
  :global(.dnd-item) {
    cursor: grab;
  }
  
  :global(.dnd-item:active) {
    cursor: grabbing;
  }
</style>
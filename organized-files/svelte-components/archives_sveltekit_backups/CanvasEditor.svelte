<script lang="ts">
  import { browser } from '$app/environment';
  import { autoTaggingMachine } from '$lib/stores/autoTaggingMachine';
  import { useMachine } from '@xstate/svelte';
  import { createEventDispatcher, onMount } from 'svelte';
  
  const dispatch = createEventDispatcher();
  const { snapshot, send } = useMachine(autoTaggingMachine);
  
  // Access state from snapshot
  $: state = $snapshot;
  
  export let caseId: string | null = null;
  export let readOnly = false;
  
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let canvasContainer: HTMLDivElement;
  
  // Enhanced file nodes with connections
  let fileNodes: Array<{
    id: string;
    name: string;
    type: string;
    content: string;
    x: number;
    y: number;
    width: number;
    height: number;
    aiTags?: unknown;
    metadata?: unknown;
    connections?: string[]; // Connected node IDs
  }> = [];
  
  // Node connections for relationship visualization
  let nodeConnections: Array<{
    fromId: string;
    toId: string;
    type: 'person' | 'location' | 'organization' | 'temporal' | 'custom';
    strength: number;
    label?: string;
  }> = [];
  
  let selectedNodeId: string | null = null;
  let hoveredNodeId: string | null = null;
  let isDragging = false;
  let isConnecting = false;
  let connectingFromId: string | null = null;
  let dragOffset = { x: 0, y: 0 };
  
  // Enhanced canvas state with zoom and pan
  let canvasWidth = 800;
  let canvasHeight = 600;
  let zoomLevel = 1;
  let minZoom = 0.1;
  let maxZoom = 3;
  let panOffset = { x: 0, y: 0 };
  let isPanning = false;
  let lastPanPoint = { x: 0, y: 0 };
  
  // Auto-save state
  let autoSaveTimer: ReturnType<typeof setInterval>;
  let isAutoSaving = false;
  
  onMount(() => {
    if (!browser) return;
    
    ctx = canvas.getContext('2d')!;
    resizeCanvas();
    draw();
    
    // Setup event listeners
    window.addEventListener('resize', resizeCanvas);
    
    // Auto-save every 10 seconds
    autoSaveTimer = setInterval(autoSave, 10000);
    
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      clearInterval(autoSaveTimer);
    };
  });
  
  function resizeCanvas() {
    if (!canvasContainer || !canvas) return;
    
    const rect = canvasContainer.getBoundingClientRect();
    canvasWidth = rect.width;
    canvasHeight = rect.height;
    
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    
    draw();
  }
  
  function draw() {
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    
    // Save context for transformations
    ctx.save();
    
    // Apply zoom and pan transformations
    ctx.translate(panOffset.x, panOffset.y);
    ctx.scale(zoomLevel, zoomLevel);
    
    // Set background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(-panOffset.x / zoomLevel, -panOffset.y / zoomLevel, canvasWidth / zoomLevel, canvasHeight / zoomLevel);
    
    // Draw grid
    drawGrid();
    
    // Draw node connections first (behind nodes)
    drawConnections();
    
    // Draw file nodes
    fileNodes.forEach(node => {
      drawFileNode(node);
    });
    
    // Draw connection preview if connecting
    if (isConnecting && connectingFromId) {
      drawConnectionPreview();
    }
    
    // Restore context
    ctx.restore();
    
    // Draw UI overlay (zoom controls, etc.)
    drawUIOverlay();
  }
  
  function drawGrid() {
    const gridSize = 40;
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1 / zoomLevel;
    
    const startX = Math.floor((-panOffset.x / zoomLevel) / gridSize) * gridSize;
    const startY = Math.floor((-panOffset.y / zoomLevel) / gridSize) * gridSize;
    const endX = startX + (canvasWidth / zoomLevel) + gridSize;
    const endY = startY + (canvasHeight / zoomLevel) + gridSize;
    
    // Vertical lines
    for (let x = startX; x < endX; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, startY);
      ctx.lineTo(x, endY);
      ctx.stroke();
    }
    
    // Horizontal lines
    for (let y = startY; y < endY; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(startX, y);
      ctx.lineTo(endX, y);
      ctx.stroke();
    }
  }
  
  function drawConnections() {
    nodeConnections.forEach(connection => {
      const fromNode = fileNodes.find(n => n.id === connection.fromId);
      const toNode = fileNodes.find(n => n.id === connection.toId);
      
      if (!fromNode || !toNode) return;
      
      const fromX = fromNode.x + fromNode.width / 2;
      const fromY = fromNode.y + fromNode.height / 2;
      const toX = toNode.x + toNode.width / 2;
      const toY = toNode.y + toNode.height / 2;
      
      // Connection line
      ctx.strokeStyle = getConnectionColor(connection.type);
      ctx.lineWidth = Math.max(2, connection.strength * 4) / zoomLevel;
      ctx.setLineDash([]);
      
      ctx.beginPath();
      ctx.moveTo(fromX, fromY);
      ctx.lineTo(toX, toY);
      ctx.stroke();
      
      // Arrow head
      drawArrowHead(fromX, fromY, toX, toY);
      
      // Connection label
      if (connection.label) {
        const midX = (fromX + toX) / 2;
        const midY = (fromY + toY) / 2;
        
        ctx.fillStyle = '#374151';
        ctx.font = `${12 / zoomLevel}px -apple-system, BlinkMacSystemFont, sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(connection.label, midX, midY - 5);
      }
    });
  }
  
  function getConnectionColor(type: string): string {
    switch (type) {
      case 'person': return '#8b5cf6';
      case 'location': return '#10b981';
      case 'organization': return '#f59e0b';
      case 'temporal': return '#ef4444';
      default: return '#6b7280';
    }
  }
  
  function drawArrowHead(fromX: number, fromY: number, toX: number, toY: number) {
    const angle = Math.atan2(toY - fromY, toX - fromX);
    const arrowLength = 15 / zoomLevel;
    const arrowAngle = Math.PI / 6;
    
    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(
      toX - arrowLength * Math.cos(angle - arrowAngle),
      toY - arrowLength * Math.sin(angle - arrowAngle)
    );
    ctx.moveTo(toX, toY);
    ctx.lineTo(
      toX - arrowLength * Math.cos(angle + arrowAngle),
      toY - arrowLength * Math.sin(angle + arrowAngle)
    );
    ctx.stroke();
  }
  
  function drawConnectionPreview() {
    // Implementation for visual connection preview while connecting nodes
    // This would show a line from the selected node to the mouse cursor
  }
  
  function drawFileNode(node: unknown) {
    const isSelected = selectedNodeId === node.id;
    const isHovered = hoveredNodeId === node.id;
    
    // Enhanced node styling with gradients and shadows
    const nodeGradient = ctx.createLinearGradient(node.x, node.y, node.x, node.y + node.height);
    
    if (isSelected) {
      nodeGradient.addColorStop(0, '#3b82f6');
      nodeGradient.addColorStop(1, '#1d4ed8');
    } else if (isHovered) {
      nodeGradient.addColorStop(0, '#f3f4f6');
      nodeGradient.addColorStop(1, '#e5e7eb');
    } else {
      nodeGradient.addColorStop(0, '#ffffff');
      nodeGradient.addColorStop(1, '#f9fafb');
    }
    
    // Drop shadow
    ctx.shadowColor = 'rgba(0, 0, 0, 0.1)';
    ctx.shadowBlur = 10 / zoomLevel;
    ctx.shadowOffsetX = 2 / zoomLevel;
    ctx.shadowOffsetY = 2 / zoomLevel;
    
    // Node background
    ctx.fillStyle = nodeGradient;
    ctx.strokeStyle = isSelected ? '#1d4ed8' : '#d1d5db';
    ctx.lineWidth = (isSelected ? 2 : 1) / zoomLevel;
    
    // Draw rounded rectangle
    roundRect(ctx, node.x, node.y, node.width, node.height, 8 / zoomLevel);
    ctx.fill();
    ctx.stroke();
    
    // Reset shadow
    ctx.shadowColor = 'transparent';
    
    // File type indicator bar
    const typeColor = getFileTypeColor(node.type);
    ctx.fillStyle = typeColor;
    ctx.fillRect(node.x, node.y, node.width, 4 / zoomLevel);
    
    // File icon
    const iconSize = 20 / zoomLevel;
    const iconX = node.x + 8 / zoomLevel;
    const iconY = node.y + 10 / zoomLevel;
    
    ctx.fillStyle = typeColor;
    ctx.font = `${iconSize}px Arial`;
    ctx.fillText(getFileIcon(node.type), iconX, iconY + iconSize);
    
    // File name
    ctx.fillStyle = isSelected ? '#ffffff' : '#1f2937';
    ctx.font = `${14 / zoomLevel}px -apple-system, BlinkMacSystemFont, sans-serif`;
    ctx.textAlign = 'left';
    ctx.fillText(
      truncateText(ctx, node.name, node.width - 40 / zoomLevel),
      iconX + iconSize + 8 / zoomLevel,
      node.y + 24 / zoomLevel
    );
    
    // AI tags indicator with count
    if (node.aiTags && node.aiTags.tags?.length > 0) {
      const tagCount = node.aiTags.tags.length;
      ctx.fillStyle = '#10b981';
      ctx.font = `${12 / zoomLevel}px -apple-system, BlinkMacSystemFont, sans-serif`;
      ctx.fillText(`🏷️ ${tagCount} tags`, node.x + 8 / zoomLevel, node.y + node.height - 8 / zoomLevel);
    }
    
    // Connection points (small circles on edges for connecting)
    if (isSelected || isHovered) {
      drawConnectionPoints(node);
    }
  }
  
  function drawConnectionPoints(node: unknown) {
    const pointSize = 6 / zoomLevel;
    const points = [
      { x: node.x + node.width / 2, y: node.y }, // Top
      { x: node.x + node.width, y: node.y + node.height / 2 }, // Right
      { x: node.x + node.width / 2, y: node.y + node.height }, // Bottom
      { x: node.x, y: node.y + node.height / 2 } // Left
    ];
    
    ctx.fillStyle = '#3b82f6';
    points.forEach(point => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, pointSize, 0, 2 * Math.PI);
      ctx.fill();
    });
  }
  
  function getFileIcon(type: string): string {
    if (type.includes('image')) return '🖼️';
    if (type.includes('pdf')) return '📄';
    if (type.includes('video')) return '🎥';
    if (type.includes('audio')) return '🎵';
    if (type.includes('text')) return '📝';
    return '📁';
  }
  
  function getFileTypeColor(type: string): string {
    if (type.includes('image')) return '#ef4444';
    if (type.includes('pdf')) return '#dc2626';
    if (type.includes('video')) return '#7c3aed';
    if (type.includes('audio')) return '#059669';
    if (type.includes('text')) return '#2563eb';
    return '#6b7280';
  }
  
  function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }
  
  function truncateText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
    if (ctx.measureText(text).width <= maxWidth) return text;
    
    let truncated = text;
    while (ctx.measureText(truncated + '...').width > maxWidth && truncated.length > 0) {
      truncated = truncated.slice(0, -1);
    }
    return truncated + '...';
  }
  
  function drawUIOverlay() {
    // Reset transformations for UI overlay
    ctx.resetTransform();
    
    // Zoom controls
    const controlsX = canvasWidth - 120;
    const controlsY = 20;
    
    // Zoom in button
    drawButton(ctx, controlsX, controlsY, 40, 30, '+', zoomLevel < maxZoom);
    
    // Zoom out button  
    drawButton(ctx, controlsX + 50, controlsY, 40, 30, '-', zoomLevel > minZoom);
    
    // Zoom level display
    ctx.fillStyle = '#374151';
    ctx.font = '12px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${Math.round(zoomLevel * 100)}%`, controlsX + 70, controlsY + 45);
  }
  
  function drawButton(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, text: string, enabled: boolean) {
    ctx.fillStyle = enabled ? '#ffffff' : '#f3f4f6';
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    
    ctx.fillRect(x, y, width, height);
    ctx.strokeRect(x, y, width, height);
    
    ctx.fillStyle = enabled ? '#374151' : '#9ca3af';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(text, x + width / 2, y + height / 2 + 6);
  }
  
  // Event handlers with enhanced functionality
  async function handleDrop(event: DragEvent) {
    event.preventDefault();
    if (readOnly) return;
    
    const files = Array.from(event.dataTransfer?.files || []);
    const rect = canvas.getBoundingClientRect();
    const dropX = (event.clientX - rect.left - panOffset.x) / zoomLevel;
    const dropY = (event.clientY - rect.top - panOffset.y) / zoomLevel;
    
    for (const file of files) {
      await processDroppedFile(file, dropX, dropY);
    }
  }
  
  async function processDroppedFile(file: File, x: number, y: number) {
    const reader = new FileReader();
    
    reader.onload = async (e) => {
      const content = e.target?.result as string;
      
      const node = {
        id: crypto.randomUUID(),
        name: file.name,
        type: file.type,
        content,
        x: x - 75,
        y: y - 25,
        width: 150,
        height: 80,
        aiTags: null,
        metadata: null,
        connections: []
      };
      
      fileNodes.push(node);
      draw();
      
      // Use XState machine for auto-tagging
      send({ type: 'DROP_FILE', node });
      
      // Auto-tag the file with enhanced processing
      await autoTagFileEnhanced(node);
    };
    
    reader.readAsDataURL(file);
  }
  
  async function autoTagFileEnhanced(node: unknown) {
    try {
      const response = await fetch('/api/ai/tag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: node.content,
          fileName: node.name,
          fileType: node.type,
          enhanced: true // Request enhanced analysis
        })
      });
      
      if (response.ok) {
        const aiTags = await response.json();
        node.aiTags = aiTags;
        
        // Auto-create connections based on shared entities
        createSmartConnections(node);
        
        draw();
        
        // Dispatch events for other panels
        dispatch('nodeUpdate', { node, aiTags });
        dispatch('nodeSelect', node);
        
        // Auto-save after tagging
        await autoSave();
      }
    } catch (error) {
      console.error('Auto-tagging failed:', error);
    }
  }
  
  function createSmartConnections(newNode: unknown) {
    if (!newNode.aiTags) return;
    
    const { people, locations, organizations } = newNode.aiTags;
    
    fileNodes.forEach(existingNode => {
      if (existingNode.id === newNode.id || !existingNode.aiTags) return;
      
      // Check for shared people
      const sharedPeople = people?.filter(person => 
        existingNode.aiTags.people?.includes(person)
      ) || [];
      
      // Check for shared locations  
      const sharedLocations = locations?.filter(location =>
        existingNode.aiTags.locations?.includes(location)
      ) || [];
      
      // Check for shared organizations
      const sharedOrganizations = organizations?.filter(org =>
        existingNode.aiTags.organizations?.includes(org)
      ) || [];
      
      // Create connections based on shared entities
      if (sharedPeople.length > 0) {
        nodeConnections.push({
          fromId: newNode.id,
          toId: existingNode.id,
          type: 'person',
          strength: sharedPeople.length / Math.max(people?.length || 1, 1),
          label: sharedPeople[0]
        });
      }
      
      if (sharedLocations.length > 0) {
        nodeConnections.push({
          fromId: newNode.id,
          toId: existingNode.id,
          type: 'location',
          strength: sharedLocations.length / Math.max(locations?.length || 1, 1),
          label: sharedLocations[0]
        });
      }
      
      if (sharedOrganizations.length > 0) {
        nodeConnections.push({
          fromId: newNode.id,
          toId: existingNode.id,
          type: 'organization',
          strength: sharedOrganizations.length / Math.max(organizations?.length || 1, 1),
          label: sharedOrganizations[0]
        });
      }
    });
  }
  
  function getMousePosition(event: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left - panOffset.x) / zoomLevel,
      y: (event.clientY - rect.top - panOffset.y) / zoomLevel
    };
  }
  
  function handleCanvasClick(event: MouseEvent) {
    const mouse = getMousePosition(event);
    
    // Check for zoom control clicks first
    const controlsX = canvasWidth - 120;
    const controlsY = 20;
    
    if (event.clientX >= controlsX && event.clientX <= controlsX + 40 && 
        event.clientY >= controlsY && event.clientY <= controlsY + 30) {
      zoomIn();
      return;
    }
    
    if (event.clientX >= controlsX + 50 && event.clientX <= controlsX + 90 && 
        event.clientY >= controlsY && event.clientY <= controlsY + 30) {
      zoomOut();
      return;
    }
    
    // Find clicked node
    const clickedNode = fileNodes.find(node => 
      mouse.x >= node.x && mouse.x <= node.x + node.width &&
      mouse.y >= node.y && mouse.y <= node.y + node.height
    );
    
    if (clickedNode) {
      selectedNodeId = clickedNode.id;
      send({ type: 'SELECT_NODE', node: clickedNode });
      dispatch('nodeSelect', clickedNode);
    } else {
      selectedNodeId = null;
      dispatch('nodeSelect', null);
    }
    
    draw();
  }
  
  function handleMouseDown(event: MouseEvent) {
    const mouse = getMousePosition(event);
    
    if (event.button === 1 || (event.button === 0 && event.ctrlKey)) {
      // Middle mouse or Ctrl+click for panning
      isPanning = true;
      lastPanPoint = { x: event.clientX, y: event.clientY };
      canvas.style.cursor = 'grabbing';
      return;
    }
    
    const clickedNode = fileNodes.find(node => 
      mouse.x >= node.x && mouse.x <= node.x + node.width &&
      mouse.y >= node.y && mouse.y <= node.y + node.height
    );
    
    if (clickedNode && !readOnly) {
      if (event.shiftKey) {
        // Shift+click to start connecting
        startConnection(clickedNode.id);
      } else {
        // Regular drag
        isDragging = true;
        selectedNodeId = clickedNode.id;
        dragOffset.x = mouse.x - clickedNode.x;
        dragOffset.y = mouse.y - clickedNode.y;
        canvas.style.cursor = 'grabbing';
      }
    }
  }
  
  function handleMouseMove(event: MouseEvent) {
    const mouse = getMousePosition(event);
    
    if (isPanning) {
      const deltaX = event.clientX - lastPanPoint.x;
      const deltaY = event.clientY - lastPanPoint.y;
      
      panOffset.x += deltaX;
      panOffset.y += deltaY;
      
      lastPanPoint = { x: event.clientX, y: event.clientY };
      draw();
      return;
    }
    
    if (isDragging && selectedNodeId) {
      const selectedNode = fileNodes.find(node => node.id === selectedNodeId);
      if (selectedNode) {
        selectedNode.x = mouse.x - dragOffset.x;
        selectedNode.y = mouse.y - dragOffset.y;
        draw();
      }
      return;
    }
    
    // Update hover state
    const hoveredNode = fileNodes.find(node => 
      mouse.x >= node.x && mouse.x <= node.x + node.width &&
      mouse.y >= node.y && mouse.y <= node.y + node.height
    );
    
    const newHoveredId = hoveredNode?.id || null;
    if (newHoveredId !== hoveredNodeId) {
      hoveredNodeId = newHoveredId;
      canvas.style.cursor = hoveredNodeId ? 'pointer' : 'default';
      draw();
    }
  }
  
  function handleMouseUp() {
    isPanning = false;
    isDragging = false;
    canvas.style.cursor = 'default';
    
    if (isDragging) {
      // Auto-save after moving nodes
      autoSave();
    }
  }
  
  function handleWheel(event: WheelEvent) {
    event.preventDefault();
    
    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(minZoom, Math.min(maxZoom, zoomLevel * zoomFactor));
    
    if (newZoom !== zoomLevel) {
      const rect = canvas.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;
      
      // Zoom towards mouse position
      const zoomRatio = newZoom / zoomLevel;
      panOffset.x = mouseX - (mouseX - panOffset.x) * zoomRatio;
      panOffset.y = mouseY - (mouseY - panOffset.y) * zoomRatio;
      
      zoomLevel = newZoom;
      draw();
    }
  }
  
  function zoomIn() {
    const newZoom = Math.min(maxZoom, zoomLevel * 1.2);
    if (newZoom !== zoomLevel) {
      zoomLevel = newZoom;
      draw();
    }
  }
  
  function zoomOut() {
    const newZoom = Math.max(minZoom, zoomLevel / 1.2);
    if (newZoom !== zoomLevel) {
      zoomLevel = newZoom;
      draw();
    }
  }
  
  function startConnection(nodeId: string) {
    isConnecting = true;
    connectingFromId = nodeId;
    canvas.style.cursor = 'crosshair';
  }
  
  function handleDragOver(event: DragEvent) {
    event.preventDefault();
  }
  
  // Auto-save functionality
  async function autoSave() {
    if (isAutoSaving) return;
    
    isAutoSaving = true;
    try {
      const canvasState = {
        nodes: fileNodes,
        connections: nodeConnections,
        viewport: { zoomLevel, panOffset },
        caseId,
        lastModified: new Date().toISOString()
      };
      
      // Save to API
      await fetch('/api/evidence/save-node', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          action: 'auto_save',
          data: { canvasState, caseId } 
        })
      });
      
      dispatch('autoSaved', canvasState);
    } catch (error) {
      console.error('Auto-save failed:', error);
    } finally {
      isAutoSaving = false;
    }
  }
  
  // Public methods
  export function addFileNode(file: unknown, x: number, y: number) {
    const node = {
      id: crypto.randomUUID(),
      name: file.name,
      type: file.type || 'application/octet-stream',
      content: file.content || '',
      x,
      y,
      width: 150,
      height: 80,
      aiTags: null,
      metadata: null,
      connections: []
    };
    
    fileNodes.push(node);
    draw();
    return node;
  }
  
  export function getSelectedNode() {
    return fileNodes.find(node => node.id === selectedNodeId) || null;
  }
  
  export function updateNode(nodeId: string, updates: unknown) {
    const node = fileNodes.find(n => n.id === nodeId);
    if (node) {
      Object.assign(node, updates);
      draw();
    }
  }
  
  export function exportCanvasState() {
    return {
      nodes: fileNodes,
      connections: nodeConnections,
      viewport: { zoomLevel, panOffset }
    };
  }
  
  export function loadCanvasState(state: unknown) {
    if (state.nodes) fileNodes = state.nodes;
    if (state.connections) nodeConnections = state.connections;
    if (state.viewport) {
      zoomLevel = state.viewport.zoomLevel || 1;
      panOffset = state.viewport.panOffset || { x: 0, y: 0 };
    }
    draw();
  }
</script>

<div class="mx-auto px-4 max-w-7xl" bind:this={canvasContainer}>
  <canvas
    bind:this={canvas}
    class="mx-auto px-4 max-w-7xl"
    on:drop={handleDrop}
    on:dragover={handleDragOver}
    onclick={handleCanvasClick}
    onmousedown={handleMouseDown}
    on:mousemove={handleMouseMove}
    onmouseup={handleMouseUp}
    onmouseleave={handleMouseUp}
    on:wheel={handleWheel}
    on:contextmenu|preventDefault
  ></canvas>
  
  <!-- Enhanced drop zone overlay -->
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">🎯</div>
      <div class="mx-auto px-4 max-w-7xl">Visual Evidence Canvas</div>
      <div class="mx-auto px-4 max-w-7xl">Drop files • AI auto-tagging • Smart connections</div>
      <div class="mx-auto px-4 max-w-7xl">Shift+click to connect • Wheel to zoom • Ctrl+drag to pan</div>
    </div>
  </div>
  
  <!-- Auto-save indicator -->
  {#if isAutoSaving}
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl"></div>
      Auto-saving...
    </div>
  {/if}
  
  <!-- XState status indicator -->
  {#if state && state.matches('processing')}
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl"></div>
      AI analyzing evidence...
    </div>
  {/if}
  
  {#if state && state.matches('error')}
    <div class="mx-auto px-4 max-w-7xl">
      AI analysis failed - Click to retry
    </div>
  {/if}
</div>

<style>
  .enhanced-canvas-editor {
    background: 
      radial-gradient(circle at 25% 25%, #f0f9ff 0%, transparent 50%),
      radial-gradient(circle at 75% 75%, #f0fdf4 0%, transparent 50%),
      linear-gradient(45deg, #f8fafc 25%, transparent 25%), 
      linear-gradient(-45deg, #f8fafc 25%, transparent 25%), 
      linear-gradient(45deg, transparent 75%, #f8fafc 75%), 
      linear-gradient(-45deg, transparent 75%, #f8fafc 75%);
    background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px, 40px 40px, 40px 40px;
    background-position: 0 0, 0 0, 0 0, 0 20px, 20px -20px, -20px 0px;
  }
</style>
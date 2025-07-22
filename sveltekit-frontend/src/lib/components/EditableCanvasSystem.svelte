<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import type { EditableNode, Evidence, CanvasState } from './types';

  // Props
  export let userId: string;
  export let canvasId: string | null = null;

  // Stores
  const canvas = writable<CanvasState | null>(null);
  const evidence = writable<Evidence[]>([]);
  const selectedNode = writable<EditableNode | null>(null);
  const isOnline = writable(true);
  const editingSessions = writable<Map<string, string>>(new Map());

  // System state
  let systemReady = false;

  // WebSocket connection
  let ws: WebSocket | null = null;

  // Canvas element reference
  let canvasElement: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;

  // Drag and drop state
  let draggedNode: EditableNode | null = null;
  let dragOffset = { x: 0, y: 0 };

  // Auto-save timeout
  let autoSaveTimeout: number;

  function initializeWebSocket() {
    try {
      // Use relative WebSocket URL for better deployment compatibility
      const wsProtocol = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = typeof window !== 'undefined' ? `${wsProtocol}//${window.location.host}/ws` : 'ws://localhost:8080';
      
      ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        isOnline.set(true);
        systemReady = true;
        
        // Join canvas room if we have a canvas
        if (canvasId) {
          ws?.send(JSON.stringify({
            type: 'JOIN_ROOM',
            room: `canvas:${canvasId}`,
            userId
          }));
        }
      };

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleRealtimeMessage(message);
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        isOnline.set(false);
        
        // Attempt to reconnect after 3 seconds
        setTimeout(initializeWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        isOnline.set(false);
      };
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
      isOnline.set(false);
    }
  }

  function handleRealtimeMessage(message: any) {
    switch (message.type) {
      case 'NODE_CREATED':
        canvas.update(c => {
          if (c) {
            c.nodes.push(message.node);
          }
          return c;
        });
        renderCanvas();
        break;
      case 'NODE_UPDATED':
        canvas.update(c => {
          if (c) {
            const index = c.nodes.findIndex(n => n.id === message.node.id);
            if (index !== -1) {
              c.nodes[index] = message.node;
            }
          }
          return c;
        });
        renderCanvas();
        break;
    }
  }

  onMount(() => {
    // Initialize canvas system
    initializeWebSocket();
    
    // Initialize canvas if element exists
    if (canvasElement) {
      ctx = canvasElement.getContext('2d')!;
      renderCanvas();
    }
  });

  onDestroy(() => {
    if (ws) {
      ws.close();
    }
  });

  // Initialize canvas when element is available
  function initializeCanvas() {
    if (canvasElement) {
      ctx = canvasElement.getContext('2d')!;
      canvas.set({
        id: canvasId || Date.now().toString(),
        nodes: [],
        connections: []
      });
      renderCanvas();
    }
  }

  function renderCanvas() {
    if (!ctx || !canvasElement) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Get current canvas state
    const currentCanvas = canvas;
    if (!currentCanvas) return;

    // Subscribe to canvas state and render
    currentCanvas.subscribe(canvasState => {
      if (!canvasState) return;
      
      // Render nodes
      canvasState.nodes.forEach(node => {
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(node.x, node.y, node.width, node.height);
        
        ctx.strokeStyle = '#ccc';
        ctx.strokeRect(node.x, node.y, node.width, node.height);
        
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.fillText(node.content, node.x + 10, node.y + 20);
      });
    });
  }

  function createNode(x: number, y: number) {
    const newNode: EditableNode = {
      id: Date.now().toString(),
      x,
      y,
      width: 200,
      height: 100,
      content: 'New Node',
      type: 'text'
    };

    canvas.update(c => {
      if (c) {
        c.nodes.push(newNode);
      }
      return c;
    });

    renderCanvas();

    // Broadcast to other users
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'NODE_CREATED',
        node: newNode,
        canvasId
      }));
    }
  }

  function handleCanvasClick(event: MouseEvent) {
    const rect = canvasElement.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Create new node on double click
    if (event.detail === 2) {
      createNode(x, y);
    }
  }

  async function uploadEvidence(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('userId', userId);

    try {
      const response = await fetch('/api/evidence/upload', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const newEvidence: Evidence = await response.json();
        evidence.update(list => [...list, newEvidence]);
        console.log('Evidence uploaded:', newEvidence);
      }
    } catch (error) {
      console.error('Upload failed:', error);
    }
  }

  function handleFileDrop(event: DragEvent) {
    event.preventDefault();
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      uploadEvidence(files[0]);
    }
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
  }
</script>

<div class="canvas-container">
  <div class="toolbar">
    <button on:click={() => canvas.set({id: Date.now().toString(), nodes: [], connections: []})}>
      New Canvas
    </button>
    <span class="status" class:online={$isOnline}>
      {$isOnline ? 'Online' : 'Offline'}
    </span>
  </div>

  <canvas
    bind:this={canvasElement}
    width="800"
    height="600"
    on:click={handleCanvasClick}
    on:drop={handleFileDrop}
    on:dragover={handleDragOver}
  ></canvas>

  <div class="evidence-panel">
    <h3>Evidence</h3>
    {#each $evidence as item}
      <div class="evidence-item">
        <span>{item.filename}</span>
      </div>
    {/each}
  </div>
</div>

<style>
  .canvas-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  .toolbar {
    display: flex;
    justify-content: space-between;
    padding: 10px;
    background: #f5f5f5;
    border-bottom: 1px solid #ddd;
  }

  .status {
    color: red;
  }

  .status.online {
    color: green;
  }

  canvas {
    border: 1px solid #ccc;
    flex: 1;
    cursor: pointer;
  }

  .evidence-panel {
    width: 300px;
    padding: 10px;
    background: #f9f9f9;
    border-left: 1px solid #ddd;
  }

  .evidence-item {
    padding: 5px;
    border: 1px solid #ddd;
    margin: 5px 0;
    background: white;
  }
</style>

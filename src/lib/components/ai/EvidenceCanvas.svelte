<script lang="ts">
  
  import { onMount, onDestroy } from 'svelte';
  import { createActor } from 'xstate';
  import { evidenceCanvasMachine } from '$lib/machines/evidenceCanvasMachine';
  import type { EvidenceNode, CanvasData } from '$lib/types/evidence';
  
  // Canvas and WebGPU integration
  let canvasEl: HTMLCanvasElement;
  let fabricCanvas: unknown;
  let webgpuContext: unknown;
  let gpuDevice: GPUDevice | null = null;
  let pdfLoading = false;
  let userPrompt = '';
  let analyzing = false;
  
  // Enhanced state management
  let evidenceNodes: EvidenceNode[] = [];
  let selectedNodes: string[] = [];
  let connectionMode = false;
  let nodeRelationships: Array<{from: string, to: string, type: string}> = [];
  
  // WebGPU acceleration props
  export let useGPUAcceleration = true;
  export let enableRealTimeProcessing = true;
  export let caseId: string | undefined = undefined;
  export let userId: string | undefined = undefined;
  
  // XState machine for canvas interactions
  let canvasMachine = createActor(evidenceCanvasMachine);

  onMount(async () => {
    // Initialize WebGPU for acceleration
    if (useGPUAcceleration && 'gpu' in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          gpuDevice = await adapter.requestDevice();
          console.log('🎮 WebGPU enabled for EvidenceCanvas');
        }
      } catch (error) {
        console.warn('WebGPU initialization failed:', error);
      }
    }
    
    // Initialize Fabric.js canvas with enhanced capabilities
    const { fabric } = await import('fabric.js');
    fabricCanvas = new fabric.Canvas(canvasEl, {
      width: 1200,
      height: 800,
      backgroundColor: '#f8fafc',
      selection: true,
      preserveObjectStacking: true
    });
    
    // Setup canvas event listeners
    setupCanvasEvents();
    
    // Start XState machine
    canvasMachine.start();
    
    // Load existing evidence nodes if caseId provided
    if (caseId) {
      await loadEvidenceNodes();
    }
    
    // Initialize real-time processing connection
    if (enableRealTimeProcessing) {
      await initializeRealtimeProcessing();
    }
  });
  
  onDestroy(() => {
    canvasMachine?.stop();
    if (fabricCanvas) {
      fabricCanvas.dispose();
    }
  });
  
  function setupCanvasEvents() {
    fabricCanvas.on('selection:created', handleSelection);
    fabricCanvas.on('selection:updated', handleSelection);
    fabricCanvas.on('selection:cleared', () => selectedNodes = []);
    fabricCanvas.on('object:moving', handleNodeMovement);
    fabricCanvas.on('object:modified', handleNodeModification);
    fabricCanvas.on('path:created', handleConnectionCreated);
  }
  
  function handleSelection(e: unknown) {
    selectedNodes = e.selected.map((obj: unknown) => obj.evidenceId).filter(Boolean);
  }
  
  function handleNodeMovement(e: unknown) {
    const node = e.target;
    if (node.evidenceId) {
      // Update node position in real-time
      updateNodePosition(node.evidenceId, { x: node.left, y: node.top });
    }
  }
  
  function handleNodeModification(e: unknown) {
    const node = e.target;
    if (node.evidenceId) {
      // Save node changes
      saveNodeChanges(node.evidenceId, node);
    }
  }
  
  function handleConnectionCreated(e: unknown) {
    // Handle relationship creation between nodes
    if (selectedNodes.length === 2) {
      createNodeRelationship(selectedNodes[0], selectedNodes[1]);
    }
  }

  async function downloadPDF() {
    pdfLoading = true;
    try {
      // Dynamically import pdf-lib for SvelteKit best practice
      const { PDFDocument } = await import('pdf-lib');
      const dataUrl = canvasEl.toDataURL('image/png');
      const pdfDoc = await PDFDocument.create();
      const page = pdfDoc.addPage([canvasEl.width, canvasEl.height]);
      const pngImage = await pdfDoc.embedPng(dataUrl);
      page.drawImage(pngImage, {
        x: 0,
        y: 0,
        width: canvasEl.width,
        height: canvasEl.height,
      });
      // Accessibility metadata
      pdfDoc.setTitle('Evidence Canvas Export');
      pdfDoc.setAuthor('Legal AI System');
      pdfDoc.setSubject('Evidence Summary');
      pdfDoc.setKeywords(['evidence', 'canvas', 'legal', 'export']);
      const pdfBytes = await pdfDoc.save();
      const blob = new Blob([pdfBytes], { type: 'application/pdf' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'evidence-canvas.pdf';
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (e) {
      alert('Failed to generate PDF: ' + (e?.message || e));
    } finally {
      pdfLoading = false;
    }
  }

  async function analyzeCanvas() {
    if (!userPrompt.trim()) return;
    analyzing = true;

    try {
      const canvasData = {
        task: "enhanced_evidence_canvas_analysis",
        prompt: userPrompt,
        context: [{
          canvas_json: fabricCanvas?.toJSON() || {},
          evidence_nodes: evidenceNodes,
          node_relationships: nodeRelationships,
          selected_nodes: selectedNodes,
          objects: fabricCanvas?.getObjects().map(obj => ({
            type: obj.type,
            evidenceId: obj.evidenceId,
            position: { x: obj.left, y: obj.top },
            text: obj.text || null,
            style: {
              fill: obj.fill,
              width: obj.width,
              height: obj.height
            }
          })) || [],
          canvas_size: { width: canvasEl.width, height: canvasEl.height },
          case_id: caseId,
          user_id: userId
        }],
        instructions: "Analyze evidence canvas with node relationships and provide comprehensive legal insights",
        options: {
          analyze_layout: true,
          extract_entities: true,
          analyze_relationships: true,
          generate_summary: true,
          suggest_connections: true,
          legal_analysis: true,
          confidence_level: 0.8,
          context_window: 8192,
          use_gpu_acceleration: useGPUAcceleration
        }
      };

      console.log('🧠 Enhanced canvas analysis payload:', canvasData);

      // Send to our WebAssembly + WebGPU middleware for processing
      const response = await fetch('http://localhost:8090/wasm/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          module: 'evidence_analyzer',
          function: 'analyze_canvas',
          data: canvasData,
          gpu_accelerated: useGPUAcceleration
        })
      });

      if (!response.ok) {
        // Fallback to go-llama service
        return await fallbackToGoLlama(canvasData);
      }

      const analysisResult = await response.json();
      console.log('📊 Enhanced analysis result:', analysisResult);

      // Process and display enhanced results
      if (analysisResult.status === 'success') {
        await displayEnhancedAnalysisResults(analysisResult);
        
        // Update canvas with AI suggestions
        if (analysisResult.suggested_connections) {
          await applySuggestedConnections(analysisResult.suggested_connections);
        }
        
        // Store results for further processing
        if (typeof window !== 'undefined') {
          window.lastCanvasAnalysis = analysisResult;
        }
      } else {
        throw new Error(analysisResult.error || 'Enhanced analysis failed');
      }

    } catch (e) {
      console.error('Enhanced analysis failed:', e);
      alert(`Enhanced analysis failed: ${e.message}`);
    } finally {
      analyzing = false;
    }
  }
  
  async function fallbackToGoLlama(canvasData: unknown) {
    console.log('🔄 Falling back to go-llama service');
    const response = await fetch('http://localhost:8081/api/evidence-canvas/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(canvasData)
    });
    
    if (!response.ok) {
      throw new Error(`Fallback analysis failed: ${response.status} ${response.statusText}`);
    }
    
    const result = await response.json();
    return await displayBasicAnalysisResults(result);
  }
  
  // Enhanced evidence canvas functionality
  async function loadEvidenceNodes() {
    try {
      const response = await fetch(`/api/evidence?caseId=${caseId}`);
      const data = await response.json();
      
      evidenceNodes = data.evidence.map((ev: unknown) => ({
        id: ev.id,
        title: ev.title,
        type: ev.evidenceType,
        position: ev.canvasPosition || { x: Math.random() * 800, y: Math.random() * 600 },
        data: ev
      }));
      
      // Render nodes on canvas
      evidenceNodes.forEach(createEvidenceNode);
    } catch (error) {
      console.error('Failed to load evidence nodes:', error);
    }
  }
  
  function createEvidenceNode(node: EvidenceNode) {
    const rect = new fabric.Rect({
      left: node.position.x,
      top: node.position.y,
      width: 150,
      height: 100,
      fill: getNodeColor(node.type),
      stroke: '#2563eb',
      strokeWidth: 2,
      rx: 8,
      ry: 8,
      evidenceId: node.id
    });
    
    const text = new fabric.Text(node.title, {
      left: node.position.x + 10,
      top: node.position.y + 40,
      width: 130,
      fontSize: 12,
      fontFamily: 'Inter, sans-serif',
      fill: '#1f2937',
      textAlign: 'center',
      evidenceId: node.id
    });
    
    const group = new fabric.Group([rect, text], {
      left: node.position.x,
      top: node.position.y,
      evidenceId: node.id,
      selectable: true,
      hasControls: true
    });
    
    fabricCanvas.add(group);
  }
  
  function getNodeColor(type: string): string {
    const colors = {
      'physical': '#fef3c7',
      'digital': '#dbeafe', 
      'document': '#f3e8ff',
      'photo': '#dcfce7',
      'video': '#fce7f3',
      'audio': '#fef9c3',
      'testimony': '#e0f2fe'
    };
    return colors[type] || '#f3f4f6';
  }
  
  async function updateNodePosition(evidenceId: string, position: {x: number, y: number}) {
    try {
      await fetch(`/api/evidence/save-node`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evidenceId,
          canvasPosition: position,
          caseId
        })
      });
    } catch (error) {
      console.error('Failed to update node position:', error);
    }
  }
  
  async function saveNodeChanges(evidenceId: string, node: unknown) {
    try {
      await fetch(`/api/evidence/save-node`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evidenceId,
          canvasPosition: { x: node.left, y: node.top },
          canvasData: node.toJSON(),
          caseId
        })
      });
    } catch (error) {
      console.error('Failed to save node changes:', error);
    }
  }
  
  async function createNodeRelationship(fromId: string, toId: string) {
    const relationshipType = prompt('Enter relationship type:', 'related_to');
    if (!relationshipType) return;
    
    try {
      const response = await fetch('/api/evidence/relationships', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fromEvidenceId: fromId,
          toEvidenceId: toId,
          relationshipType,
          caseId
        })
      });
      
      if (response.ok) {
        nodeRelationships.push({ from: fromId, to: toId, type: relationshipType });
        drawConnection(fromId, toId, relationshipType);
      }
    } catch (error) {
      console.error('Failed to create relationship:', error);
    }
  }
  
  function drawConnection(fromId: string, toId: string, type: string) {
    const fromNode = fabricCanvas.getObjects().find((obj: unknown) => obj.evidenceId === fromId);
    const toNode = fabricCanvas.getObjects().find((obj: unknown) => obj.evidenceId === toId);
    
    if (fromNode && toNode) {
      const line = new fabric.Line([
        fromNode.left + fromNode.width / 2,
        fromNode.top + fromNode.height / 2,
        toNode.left + toNode.width / 2,
        toNode.top + toNode.height / 2
      ], {
        stroke: getConnectionColor(type),
        strokeWidth: 2,
        strokeDashArray: type === 'weak' ? [5, 5] : null,
        selectable: false,
        evented: false
      });
      
      fabricCanvas.add(line);
      fabricCanvas.sendToBack(line);
    }
  }
  
  function getConnectionColor(type: string): string {
    const colors = {
      'related_to': '#3b82f6',
      'contradicts': '#ef4444',
      'supports': '#10b981',
      'sequence': '#f59e0b',
      'weak': '#9ca3af'
    };
    return colors[type] || '#6b7280';
  }
  
  async function initializeRealtimeProcessing() {
    try {
      // Connect to WebSocket for real-time updates
      const ws = new WebSocket(`ws://localhost:8090/ws`);
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'evidence_update' && data.caseId === caseId) {
          handleRealtimeEvidenceUpdate(data);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket connection failed:', error);
      };
    } catch (error) {
      console.error('Failed to initialize real-time processing:', error);
    }
  }
  
  function handleRealtimeEvidenceUpdate(data: unknown) {
    // Update canvas in real-time when evidence is modified
    const existingNode = fabricCanvas.getObjects().find((obj: unknown) => obj.evidenceId === data.evidenceId);
    if (existingNode) {
      // Update existing node
      existingNode.set(data.updates);
      fabricCanvas.renderAll();
    } else if (data.action === 'created') {
      // Add new node
      const newNode = {
        id: data.evidenceId,
        title: data.title,
        type: data.type,
        position: data.position || { x: Math.random() * 800, y: Math.random() * 600 },
        data: data
      };
      createEvidenceNode(newNode);
    }
  }
  
  async function displayEnhancedAnalysisResults(result: unknown) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
      <div class="bg-white rounded-lg p-6 max-w-4xl max-h-[80vh] overflow-y-auto">
        <h3 class="text-xl font-bold mb-4">🧠 Enhanced Evidence Analysis</h3>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 class="font-semibold mb-2">📊 Analysis Summary</h4>
            <p class="text-gray-700">${result.summary}</p>
            
            <h4 class="font-semibold mt-4 mb-2">🎯 Confidence Score</h4>
            <div class="w-full bg-gray-200 rounded-full h-2">
              <div class="bg-blue-600 h-2 rounded-full" style="width: ${(result.confidence * 100)}%"></div>
            </div>
            <span class="text-sm text-gray-600">${(result.confidence * 100).toFixed(1)}%</span>
          </div>
          
          <div>
            <h4 class="font-semibold mb-2">🔍 Entities Found</h4>
            <ul class="list-disc list-inside text-sm text-gray-700">
              ${(result.entities || []).map((entity: unknown) => `<li>${entity.text} (${entity.type})</li>`).join('')}
            </ul>
            
            <h4 class="font-semibold mt-4 mb-2">🔗 Suggested Connections</h4>
            <ul class="list-disc list-inside text-sm text-gray-700">
              ${(result.suggested_connections || []).map((conn: unknown) => `<li>${conn.from} → ${conn.to} (${conn.type})</li>`).join('')}
            </ul>
          </div>
        </div>
        
        ${result.legal_insights ? `
          <div class="mt-6 p-4 bg-blue-50 rounded-lg">
            <h4 class="font-semibold mb-2">⚖️ Legal Insights</h4>
            <p class="text-gray-700">${result.legal_insights}</p>
          </div>
        ` : ''}
        
        <div class="flex justify-end mt-6 space-x-3">
          <button onclick="this.closest('.fixed').remove()" class="px-4 py-2 bg-gray-300 rounded hover:bg-gray-400">Close</button>
          <button onclick="applySuggestions(${JSON.stringify(result).replace(/"/g, '&quot;')})" class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">Apply Suggestions</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
  }
  
  async function displayBasicAnalysisResults(result: unknown) {
    alert(`Analysis Complete!\\n\\nSummary: ${result.summary}\\n\\nConfidence: ${(result.confidence * 100).toFixed(1)}%\\n\\nEntities found: ${result.entities?.length || 0}`);
  }
  
  async function applySuggestedConnections(suggestions: unknown[]) {
    for (const suggestion of suggestions) {
      if (confirm(`Create connection: ${suggestion.from} → ${suggestion.to} (${suggestion.type})?`)) {
        await createNodeRelationship(suggestion.from, suggestion.to);
      }
    }
  }
  
  function toggleConnectionMode() {
    connectionMode = !connectionMode;
    fabricCanvas.defaultCursor = connectionMode ? 'crosshair' : 'default';
  }
  
  function exportCanvasData(): CanvasData {
    return {
      canvas_json: fabricCanvas.toJSON(),
      evidence_nodes: evidenceNodes,
      node_relationships: nodeRelationships,
      case_id: caseId,
      user_id: userId,
      timestamp: new Date().toISOString()
    };
  }
  
  async function saveCanvasState() {
    try {
      const canvasData = exportCanvasData();
      await fetch('/api/canvas/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(canvasData)
      });
      alert('Canvas saved successfully!');
    } catch (error) {
      console.error('Failed to save canvas:', error);
      alert('Failed to save canvas');
    }
  }

</script>

    <div class="evidence-canvas">
      <!-- Enhanced Control Panel -->
      <div class="controls-panel bg-white rounded-lg shadow-sm border p-4 mb-4">
        <div class="flex flex-wrap items-center gap-3 mb-3">
          <!-- AI Analysis Section -->
          <div class="flex-1 min-w-[300px]">
            <input
              bind:value={userPrompt}
              placeholder="Ask the AI to analyze evidence relationships, find patterns, or suggest connections..."
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              onkeydown={(e) => e.key === 'Enter' && analyzeCanvas()}
            />
          </div>
          
          <button
            onclick={analyzeCanvas}
            class="px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-md hover:from-indigo-700 hover:to-purple-700 transition-all duration-200 flex items-center space-x-2"
            disabled={analyzing}
            aria-busy={analyzing}
          >
            {#if analyzing}
              <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Analyzing...</span>
            {:else}
              <span>🧠</span>
              <span>Analyze with AI</span>
            {/if}
          </button>
        </div>
        
        <!-- Canvas Controls -->
        <div class="flex flex-wrap items-center gap-2">
          <button
            onclick={toggleConnectionMode}
            class="px-3 py-2 {connectionMode ? 'bg-green-600 text-white' : 'bg-gray-100 text-gray-700'} rounded-md hover:opacity-80 transition-opacity flex items-center space-x-1"
          >
            <span>🔗</span>
            <span>{connectionMode ? 'Exit Connection Mode' : 'Connect Evidence'}</span>
          </button>
          
          <button
            onclick={() => loadEvidenceNodes()}
            class="px-3 py-2 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors flex items-center space-x-1"
          >
            <span>📁</span>
            <span>Load Evidence</span>
          </button>
          
          <button
            onclick={saveCanvasState}
            class="px-3 py-2 bg-green-100 text-green-700 rounded-md hover:bg-green-200 transition-colors flex items-center space-x-1"
          >
            <span>💾</span>
            <span>Save Canvas</span>
          </button>
          
          <button
            onclick={downloadPDF}
            class="px-3 py-2 bg-orange-100 text-orange-700 rounded-md hover:bg-orange-200 transition-colors flex items-center space-x-1"
            disabled={pdfLoading}
            aria-busy={pdfLoading}
          >
            {#if pdfLoading}
              <div class="w-4 h-4 border-2 border-orange-600 border-t-transparent rounded-full animate-spin"></div>
              <span>Generating...</span>
            {:else}
              <span>📄</span>
              <span>Export PDF</span>
            {/if}
          </button>
          
          <!-- GPU Status Indicator -->
          {#if useGPUAcceleration}
            <div class="flex items-center space-x-2 px-3 py-2 bg-purple-50 rounded-md">
              <div class="w-2 h-2 {gpuDevice ? 'bg-green-500' : 'bg-red-500'} rounded-full"></div>
              <span class="text-sm font-medium {gpuDevice ? 'text-green-700' : 'text-red-700'}">
                {gpuDevice ? '🚀 GPU Accelerated' : '⚠️ GPU Failed'}
              </span>
            </div>
          {/if}
        </div>
      </div>

      <!-- Canvas Status Bar -->
      {#if evidenceNodes.length > 0 || nodeRelationships.length > 0}
        <div class="status-bar bg-gray-50 rounded-lg p-3 mb-4 flex items-center justify-between text-sm">
          <div class="flex items-center space-x-4">
            <span class="flex items-center space-x-1">
              <span class="w-3 h-3 bg-blue-500 rounded"></span>
              <span>{evidenceNodes.length} Evidence Items</span>
            </span>
            <span class="flex items-center space-x-1">
              <span class="w-3 h-3 bg-green-500 rounded"></span>
              <span>{nodeRelationships.length} Connections</span>
            </span>
            <span class="flex items-center space-x-1">
              <span class="w-3 h-3 bg-purple-500 rounded"></span>
              <span>{selectedNodes.length} Selected</span>
            </span>
          </div>
          
          {#if caseId}
            <span class="text-gray-600">Case: {caseId.substring(0, 8)}...</span>
          {/if}
        </div>
      {/if}

      <!-- Enhanced Canvas Wrapper -->
      <div class="evidence-canvas-wrapper">
        <div class="canvas-container">
          <canvas bind:this={canvasEl} width="1200" height="800"></canvas>
          
          <!-- Canvas Overlay for Connection Mode -->
          {#if connectionMode}
            <div class="canvas-overlay">
              <div class="connection-help">
                <p>🔗 Connection Mode Active</p>
                <p>Select two evidence items to create a relationship</p>
              </div>
            </div>
          {/if}
        </div>
      </div>

      <!-- Evidence Legend -->
      {#if evidenceNodes.length > 0}
        <div class="legend bg-white rounded-lg shadow-sm border p-4 mt-4">
          <h4 class="font-semibold mb-3">Evidence Types</h4>
          <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3 text-sm">
            <div class="flex items-center space-x-2">
              <div class="w-4 h-4 rounded" style="background-color: #fef3c7"></div>
              <span>Physical</span>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-4 h-4 rounded" style="background-color: #dbeafe"></div>
              <span>Digital</span>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-4 h-4 rounded" style="background-color: #f3e8ff"></div>
              <span>Document</span>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-4 h-4 rounded" style="background-color: #dcfce7"></div>
              <span>Photo</span>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-4 h-4 rounded" style="background-color: #fce7f3"></div>
              <span>Video</span>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-4 h-4 rounded" style="background-color: #fef9c3"></div>
              <span>Audio</span>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-4 h-4 rounded" style="background-color: #e0f2fe"></div>
              <span>Testimony</span>
            </div>
          </div>
        </div>
      {/if}
    </div>

    <style>
      .evidence-canvas {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        max-width: 100%;
      }
      
      .controls-panel {
        /* Enhanced control panel styling */
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      }
      
      .status-bar {
        /* Status bar with subtle styling */
        border-left: 4px solid #3b82f6;
      }
      
      .evidence-canvas-wrapper {
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      }
      
      .canvas-container {
        position: relative;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
      }
      
      canvas {
        background: #ffffff;
        border-radius: 8px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        max-width: 100%;
        height: auto;
      }
      
      .canvas-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(59, 130, 246, 0.1);
        backdrop-filter: blur(1px);
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: none;
        z-index: 10;
      }
      
      .connection-help {
        background: rgba(59, 130, 246, 0.9);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      }
      
      .connection-help p {
        margin: 0;
      }
      
      .connection-help p:first-child {
        font-size: 1.1rem;
        margin-bottom: 0.25rem;
      }
      
      .connection-help p:last-child {
        font-size: 0.9rem;
        opacity: 0.9;
      }
      
      .legend {
        border-left: 4px solid #8b5cf6;
      }
      
      /* Responsive design */
      @media (max-width: 768px) {
        .controls-panel .flex {
          flex-direction: column;
          align-items: stretch;
        }
        
        .controls-panel .flex > div:first-child {
          margin-bottom: 1rem;
        }
        
        .evidence-canvas-wrapper {
          height: 400px;
        }
        
        canvas {
          width: 100%;
          max-width: none;
        }
        
        .legend .grid {
          grid-template-columns: repeat(2, 1fr);
        }
      }
      
      @media (max-width: 480px) {
        .legend .grid {
          grid-template-columns: 1fr;
        }
      }
      
      /* Animation for nodes */
      .evidence-node {
        transition: all 0.2s ease-in-out;
      }
      
      .evidence-node:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
      }
      
      /* Connection line animations */
      .evidence-connection {
        transition: stroke-width 0.2s ease-in-out;
      }
      
      .evidence-connection:hover {
        stroke-width: 3;
      }
    </style>


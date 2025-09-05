<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { 
    Upload, Move, RotateCcw, Trash2, ZoomIn, ZoomOut, 
    Save, Download, Image as ImageIcon, FileText 
  } from 'lucide-svelte';

  interface Props {
    width?: number;
    height?: number;
    caseId?: string;
    readOnly?: boolean;
    onSave?: (data: { objects: any[] }) => void;
    onDelete?: (data: { objectId: string }) => void;
    onSelect?: (data: { object: any }) => void;
  }

  let {
    width = 800,
    height = 600,
    caseId = undefined,
    readOnly = false,
    onSave,
    onDelete,
    onSelect
  } = $props();

  // Fabric.js canvas instance
  let fabricCanvas: any = null;
  let canvasElement: HTMLCanvasElement
  let isLoading = false;
  let selectedObject: any = null;
  let canvasObjects: any[] = [];
  let zoomLevel = 1;

  // Evidence management
  interface EvidenceItem {
    id: string
    type: 'image' | 'document' | 'annotation';
    title: string
    url?: string;
    x: number
    y: number
    metadata: Record<string, any>;
  }

  let evidenceItems: EvidenceItem[] = [];

  onMount(async () => {
    // Dynamically import Fabric.js to avoid SSR issues
    const fabric = await import('fabric');
    
    fabricCanvas = new fabric.Canvas(canvasElement, {
      width,
      height,
      backgroundColor: '#f8fafc',
      selection: !readOnly
    });

    setupCanvasEvents();
    loadCanvasData();
  });

  function setupCanvasEvents() {
    if (!fabricCanvas) return;

    // Object selection
    fabricCanvas.on('selection:created', (e: any) => {
      selectedObject = e.selected[0];
      onSelect?.({ object: selectedObject });
    });

    fabricCanvas.on('selection:cleared', () => {
      selectedObject = null;
    });

    // Object modification
    fabricCanvas.on('object:modified', () => {
      updateCanvasObjects();
    });

    // Mouse wheel zoom
    fabricCanvas.on('mouse:wheel', (opt: any) => {
      const delta = opt.e.deltaY;
      let zoom = fabricCanvas.getZoom();
      zoom *= 0.999 ** delta;
      
      if (zoom > 20) zoom = 20;
      if (zoom < 0.01) zoom = 0.01;
      
      fabricCanvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
      zoomLevel = zoom;
      opt.e.preventDefault();
      opt.e.stopPropagation();
    });
  }

  async function loadCanvasData() {
    if (!caseId) return;
    
    try {
      isLoading = true;
      // Load evidence items for this case
      const response = await fetch(`/api/cases/${caseId}/evidence`);
      const evidence = await response.json();
      evidenceItems = evidence;
      
      // Add evidence to canvas
      for (const item of evidenceItems) {
        await addEvidenceToCanvas(item);
      }
    } catch (error) {
      console.error('Failed to load canvas data:', error);
    } finally {
      isLoading = false;
    }
  }

  async function addEvidenceToCanvas(evidence: EvidenceItem) {
    if (!fabricCanvas) return;

    try {
      if (evidence.type === 'image' && evidence.url) {
        const fabric = await import('fabric');
        
        fabric.Image.fromURL(evidence.url, (img: any) => {
          img.set({
            left: evidence.x,
            top: evidence.y,
            scaleX: 0.5,
            scaleY: 0.5,
            selectable: !readOnly,
            evidenceId: evidence.id,
            evidenceType: evidence.type
          });
          
          fabricCanvas.add(img);
          fabricCanvas.renderAll();
        });
      } else if (evidence.type === 'document') {
        const fabric = await import('fabric');
        
        const rect = new fabric.Rect({
          left: evidence.x,
          top: evidence.y,
          width: 120,
          height: 160,
          fill: '#ffffff',
          stroke: '#e2e8f0',
          strokeWidth: 2,
          selectable: !readOnly,
          evidenceId: evidence.id,
          evidenceType: evidence.type
        });

        const text = new fabric.Text(evidence.title, {
          left: evidence.x + 10,
          top: evidence.y + 10,
          fontSize: 12,
          fontFamily: 'Arial',
          fill: '#1f2937',
          selectable: false
        });

        const group = new fabric.Group([rect, text], {
          left: evidence.x,
          top: evidence.y,
          selectable: !readOnly,
          evidenceId: evidence.id,
          evidenceType: evidence.type
        });

        fabricCanvas.add(group);
        fabricCanvas.renderAll();
      }
    } catch (error) {
      console.error('Failed to add evidence to canvas:', error);
    }
  }

  async function handleFileUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    const files = input.files;
    
    if (!files || files.length === 0) return;
    
    for (const file of Array.from(files)) {
      await uploadEvidence(file);
    }
    
    input.value = ''; // Reset input
  }

  async function uploadEvidence(file: File) {
    try {
      isLoading = true;
      
      // Upload file
      const formData = new FormData();
      formData.append('file', file);
      formData.append('caseId', caseId || '');
      
      const uploadResponse = await fetch('/api/evidence/upload', {
        method: 'POST',
        body: formData
      });
      
      if (!uploadResponse.ok) {
        throw new Error('Upload failed');
      }
      
      const uploadResult = await uploadResponse.json();
      
      // Create evidence item
      const evidence: EvidenceItem = {
        id: uploadResult.id,
        type: file.type.startsWith('image/') ? 'image' : 'document',
        title: file.name,
        url: uploadResult.url,
        x: Math.random() * (width - 200) + 100,
        y: Math.random() * (height - 200) + 100,
        metadata: {
          fileSize: file.size,
          fileType: file.type,
          uploadDate: new Date().toISOString()
        }
      };
      
      evidenceItems = [...evidenceItems, evidence];
      await addEvidenceToCanvas(evidence);
      
    } catch (error) {
      console.error('Failed to upload evidence:', error);
    } finally {
      isLoading = false;
    }
  }

  function addAnnotation() {
    if (!fabricCanvas) return;

    import('fabric').then(fabric => {
      const text = new fabric.IText('Click to edit annotation', {
        left: width / 2,
        top: height / 2,
        fontSize: 16,
        fontFamily: 'Arial',
        fill: '#dc2626',
        backgroundColor: '#fef2f2',
        padding: 8,
        selectable: !readOnly,
        editable: !readOnly
      });

      fabricCanvas.add(text);
      fabricCanvas.setActiveObject(text);
      fabricCanvas.renderAll();
    });
  }

  function deleteSelected() {
    if (!fabricCanvas || !selectedObject) return;
    
    const evidenceId = selectedObject.evidenceId;
    
    fabricCanvas.remove(selectedObject);
    fabricCanvas.renderAll();
    
    if (evidenceId) {
      evidenceItems = evidenceItems.filter(item => item.id !== evidenceId);
      onDelete?.({ objectId: evidenceId });
    }
    
    selectedObject = null;
  }

  function zoomIn() {
    if (!fabricCanvas) return;
    const zoom = Math.min(fabricCanvas.getZoom() * 1.2, 5);
    fabricCanvas.setZoom(zoom);
    zoomLevel = zoom;
  }

  function zoomOut() {
    if (!fabricCanvas) return;
    const zoom = Math.max(fabricCanvas.getZoom() * 0.8, 0.1);
    fabricCanvas.setZoom(zoom);
    zoomLevel = zoom;
  }

  function resetZoom() {
    if (!fabricCanvas) return;
    fabricCanvas.setZoom(1);
    zoomLevel = 1;
  }

  function saveCanvas() {
    if (!fabricCanvas) return;
    
    const objects = fabricCanvas.getObjects();
    updateCanvasObjects();
    onSave?.({ objects: canvasObjects });
  }

  function updateCanvasObjects() {
    if (!fabricCanvas) return;
    canvasObjects = fabricCanvas.getObjects().map((obj: any) => ({
      type: obj.type,
      left: obj.left,
      top: obj.top,
      width: obj.width,
      height: obj.height,
      angle: obj.angle,
      scaleX: obj.scaleX,
      scaleY: obj.scaleY,
      evidenceId: obj.evidenceId,
      evidenceType: obj.evidenceType
    }));
  }

  function exportCanvas() {
    if (!fabricCanvas) return;
    
    const dataURL = fabricCanvas.toDataURL({
      format: 'png',
      quality: 1,
      multiplier: 2
    });
    
    // Create download link
    const link = document.createElement('a');
    link.download = `case-${caseId || 'canvas'}-${Date.now()}.png`;
    link.href = dataURL;
    link.click();
  }
</script>

<div class="fabric-canvas-container">
  <!-- Toolbar -->
  <Card class="mb-4">
    <CardHeader class="pb-3">
      <CardTitle class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <ImageIcon class="h-5 w-5" />
          Evidence Canvas
          {#if evidenceItems.length > 0}
            <Badge variant="secondary">{evidenceItems.length} items</Badge>
          {/if}
        </div>
        <div class="flex items-center gap-2 text-sm text-gray-600">
          Zoom: {Math.round(zoomLevel * 100)}%
        </div>
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div class="flex flex-wrap gap-2">
        <!-- File Upload -->
        {#if !readOnly}
          <label class="cursor-pointer">
            <input
              type="file"
              multiple
              accept="image/*,.pdf,.doc,.docx,.txt"
              class="hidden"
              onchange={handleFileUpload}
              disabled={isLoading}
            />
            <Button variant="outline" disabled={isLoading}>
              <Upload class="h-4 w-4 mr-2" />
              Upload Evidence
            </Button>
          </label>
        {/if}

        <!-- Add Annotation -->
        {#if !readOnly}
          <Button variant="outline" onclick={addAnnotation}>
            <FileText class="h-4 w-4 mr-2" />
            Add Note
          </Button>
        {/if}

        <!-- Zoom Controls -->
        <Button variant="outline" onclick={zoomIn}>
          <ZoomIn class="h-4 w-4" />
        </Button>
        <Button variant="outline" onclick={zoomOut}>
          <ZoomOut class="h-4 w-4" />
        </Button>
        <Button variant="outline" onclick={resetZoom}>
          <RotateCcw class="h-4 w-4" />
        </Button>

        <!-- Object Controls -->
        {#if selectedObject && !readOnly}
          <Button variant="destructive" onclick={deleteSelected}>
            <Trash2 class="h-4 w-4 mr-2" />
            Delete
          </Button>
        {/if}

        <!-- Save & Export -->
        {#if !readOnly}
          <Button variant="default" onclick={saveCanvas}>
            <Save class="h-4 w-4 mr-2" />
            Save
          </Button>
        {/if}
        
        <Button variant="outline" onclick={exportCanvas}>
          <Download class="h-4 w-4 mr-2" />
          Export
        </Button>
      </div>
    </CardContent>
  </Card>

  <!-- Canvas -->
  <div class="canvas-wrapper relative border border-gray-200 rounded-lg overflow-hidden">
    <canvas
      bind:this={canvasElement}
      class="block"
    />
    
    {#if isLoading}
      <div class="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center">
        <div class="text-center">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
          <p class="text-sm text-gray-600">Loading...</p>
        </div>
      </div>
    {/if}
  </div>

  <!-- Object Properties Panel -->
  {#if selectedObject}
    <Card class="mt-4">
      <CardHeader>
        <CardTitle class="text-lg">Selected Object</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="grid grid-cols-2 gap-4 text-sm">
          <div>
            <label class="font-medium">Type:</label>
            <p class="text-gray-600">{selectedObject.type}</p>
          </div>
          <div>
            <label class="font-medium">Position:</label>
            <p class="text-gray-600">
              {Math.round(selectedObject.left)}, {Math.round(selectedObject.top)}
            </p>
          </div>
          <div>
            <label class="font-medium">Size:</label>
            <p class="text-gray-600">
              {Math.round(selectedObject.width * selectedObject.scaleX)} × 
              {Math.round(selectedObject.height * selectedObject.scaleY)}
            </p>
          </div>
          <div>
            <label class="font-medium">Rotation:</label>
            <p class="text-gray-600">{Math.round(selectedObject.angle)}°</p>
          </div>
          {#if selectedObject.evidenceId}
            <div class="col-span-2">
              <label class="font-medium">Evidence ID:</label>
              <p class="text-gray-600 font-mono text-xs">{selectedObject.evidenceId}</p>
            </div>
          {/if}
        </div>
      </CardContent>
    </Card>
  {/if}
</div>

<style>
  .fabric-canvas-container {
    width: 100%;
  }
  
  .canvas-wrapper {
    display: inline-block;
    background: white
  }
</style>
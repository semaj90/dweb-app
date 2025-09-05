<!-- Enhanced Canvas Evidence Board with Fabric.js Integration -->
<script lang="ts">
  interface Props {
    onevidenceUpdated?: (event?: any) => void;
    onsave?: (event?: any) => void;
  }


  import { browser } from "$app/environment";
  import Button from "$lib/components/ui/Button.svelte";
  import { notifications } from "$lib/stores/notification";
  import {
    Circle,
    Download,
    Image,
    Move,
    Redo,
    Save,
    Square,
    Trash2,
    Type,
    Undo,
    ZoomIn,
    ZoomOut,
  } from "lucide-svelte";
  
  
  // --- XState workflow state machine ---
  // Dynamically imported in onMount for SSR safety
  let canvasService: any = null;
  let currentMode: string = "evidence";
  let canvasContainer: HTMLDivElement
  let fabricCanvas: any = null;
  let selectedTool = "select";
  let isDrawing = false;
  let fabricLoaded = false;
  let canvasHistory: string[] = [];
  let historyIndex = -1;
  let zoom = 1;
  let readonly = false;
  let caseId: string | undefined = undefined;
  let evidenceItems: Array<any> = [];
  function setWorkflowMode(mode: string) {
    if (canvasService) canvasService.send(mode.toUpperCase());
}
  onMount(async () => {
    if (!browser) return;
    try {
      // Dynamically import Fabric.js
      const fabricModule = await import("fabric");
      const fabricLib = fabricModule.default;
      if (!fabricLib || !canvasContainer) return;
      // Create canvas element
      const canvasElement = document.createElement("canvas");
      canvasElement.width = 1200;
      canvasElement.height = 800;
      canvasContainer.appendChild(canvasElement);
      // Initialize Fabric canvas
      fabricCanvas = new fabricLib.Canvas(canvasElement, {
        backgroundColor: "#f8fafc",
        selection: !readonly,
        preserveObjectStacking: true,
        enableRetinaScaling: true,
      });
      fabricLoaded = true;
      // Add event listeners
      // ...existing event setup...
      // --- XState workflow state machine ---
      const xstateModule = await import("xstate");
      const { createMachine, createActor } = xstateModule;
      const canvasMachine = createMachine({
        id: "canvasWorkflow",
        initial: "evidence",
        states: {
          evidence: { on: { DRAW: "drawing", ANNOTATE: "annotation" } },
          drawing: { on: { EVIDENCE: "evidence", ANNOTATE: "annotation" } },
          annotation: { on: { EVIDENCE: "evidence", DRAW: "drawing" } },
        },
      });
      canvasService = createActor(canvasMachine);
      canvasService.subscribe((state: any) => {
        currentMode = state.value as string;
      });
      canvasService.start();
      // Load evidence items
      if (evidenceItems && evidenceItems.length) {
        for (const item of evidenceItems) {
          await addEvidenceToCanvas(item);
}
        fabricCanvas.renderAll();
}
      // Save initial state
      saveCanvasState();
    } catch (error) {
      console.error("Failed to initialize Fabric.js or XState:", error);
      notifications.add({
        type: "error",
        title: "Canvas Error",
        message:
          "Failed to initialize canvas or workflow state. Some features may not work.",
      });
}
  });

  onDestroy(() => {
    if (canvasService) canvasService.stop();
    if (fabricCanvas) fabricCanvas.dispose();
  });

  async function addEvidenceToCanvas(item: any) {
    if (!fabricCanvas) return;

    try {
      const fabricModule = await import("fabric");
      const fabricLib = fabricModule.default;

      let fabricObject: any

      if (item.type === "image" && item.thumbnailUrl) {
        // Add image
        try {
          const img = await fabricLib.FabricImage.fromURL(item.thumbnailUrl);
          img.set({
            left: item.x || 100,
            top: item.y || 100,
            scaleX: (item.width || 200) / img.width,
            scaleY: (item.height || 150) / img.height,
            selectable: !readonly,
            evented: !readonly,
          });
          fabricObject = img;
        } catch (imgError) {
          console.error("Failed to load image:", imgError);
          // Fallback to text representation
          fabricObject = new fabricLib.Textbox(`🖼️ ${item.title}`, {
            left: item.x || 100,
            top: item.y || 100,
            width: item.width || 200,
            fontSize: 14,
            fontFamily: "Arial",
            fill: "#6b7280",
            selectable: !readonly,
            evented: !readonly,
          });
}
      } else {
        // Add as text/document representation
        const text = `${getTypeIcon(item.type)} ${item.title}`;
        fabricObject = new fabricLib.Textbox(text, {
          left: item.x || 100,
          top: item.y || 100,
          width: item.width || 200,
          fontSize: 14,
          fontFamily: "Arial",
          fill: "#1f2937",
          backgroundColor: "#ffffff",
          padding: 10,
          borderColor: "#e5e7eb",
          cornerColor: "#3b82f6",
          selectable: !readonly,
          evented: !readonly,
        });
}
      // Add metadata
      fabricObject.set({
        evidenceId: item.id,
        evidenceType: item.type,
        customType: "evidence",
      });

      fabricCanvas.add(fabricObject);
    } catch (error) {
      console.error("Error adding evidence to canvas:", error);
}}
  function getTypeIcon(type: string): string {
    switch (type) {
      case "image":
        return "🖼️";
      case "document":
        return "📄";
      case "video":
        return "🎥";
      case "audio":
        return "🎵";
      default:
        return "📎";
}}
  function selectTool(tool: string) {
    selectedTool = tool;

    if (!fabricCanvas) return;

    switch (tool) {
      case "select":
        fabricCanvas.isDrawingMode = false;
        fabricCanvas.selection = true;
        break;
      case "draw":
        fabricCanvas.isDrawingMode = true;
        fabricCanvas.selection = false;
        break;
      case "text":
        fabricCanvas.isDrawingMode = false;
        fabricCanvas.selection = true;
        addTextBox();
        break;
}}
  async function addShape(shape: "rectangle" | "circle") {
    if (!fabricCanvas) return;

    try {
      const fabricModule = await import("fabric");
      const fabricLib = fabricModule.default;

      let fabricObject: any

      if (shape === "rectangle") {
        fabricObject = new fabricLib.Rect({
          left: 100,
          top: 100,
          width: 100,
          height: 80,
          fill: "rgba(59, 130, 246, 0.1)",
          stroke: "#3b82f6",
          strokeWidth: 2,
        });
      } else {
        fabricObject = new fabricLib.Circle({
          left: 100,
          top: 100,
          radius: 50,
          fill: "rgba(16, 185, 129, 0.1)",
          stroke: "#10b981",
          strokeWidth: 2,
        });
}
      fabricObject.set({
        customType: "shape",
      });

      fabricCanvas.add(fabricObject);
      fabricCanvas.setActiveObject(fabricObject);
      saveCanvasState();
    } catch (error) {
      console.error("Error adding shape:", error);
}}
  async function addTextBox() {
    if (!fabricCanvas) return;

    try {
      const fabricModule = await import("fabric");
      const fabric = fabricModule.default || fabricModule;

      const textbox = new fabric.Textbox("Type here...", {
        left: 100,
        top: 100,
        width: 200,
        fontSize: 16,
        fontFamily: "Arial",
        fill: "#1f2937",
        backgroundColor: "rgba(255, 255, 255, 0.9)",
        padding: 10,
      });

      textbox.set({
        customType: "text",
      });

      fabricCanvas.add(textbox);
      fabricCanvas.setActiveObject(textbox);
      saveCanvasState();
    } catch (error) {
      console.error("Error adding text:", error);
}}
  let currentPath: any = null;

  function startDrawing(pointer: { x: number y: number }) {
    // Drawing implementation
}
  function continueDrawing(pointer: { x: number y: number }) {
    // Continue drawing implementation
}
  function finishDrawing() {
    saveCanvasState();
}
  function saveCanvasState() {
    if (!fabricCanvas) return;

    const state = JSON.stringify(
      fabricCanvas.toJSON(["evidenceId", "evidenceType", "customType"])
    );

    // Manage history
    if (historyIndex < canvasHistory.length - 1) {
      canvasHistory = canvasHistory.slice(0, historyIndex + 1);
}
    canvasHistory.push(state);
    historyIndex++;

    // Limit history size
    if (canvasHistory.length > 50) {
      canvasHistory = canvasHistory.slice(1);
      historyIndex--;
}}
  function undo() {
    if (historyIndex > 0) {
      historyIndex--;
      loadCanvasState(canvasHistory[historyIndex]);
}}
  function redo() {
    if (historyIndex < canvasHistory.length - 1) {
      historyIndex++;
      loadCanvasState(canvasHistory[historyIndex]);
}}
  async function loadCanvasState(state: string) {
    if (!fabricCanvas) return;

    try {
      fabricCanvas.loadFromJSON(state, () => {
        fabricCanvas.renderAll();
        updateEvidencePositions();
      });
    } catch (error) {
      console.error("Error loading canvas state:", error);
}}
  function updateEvidencePositions() {
    if (!fabricCanvas) return;

    const objects = fabricCanvas.getObjects();

    objects.forEach((obj: any) => {
      if (obj.evidenceId) {
        onevidenceUpdated?.();
}
    });
}
  function zoomIn() {
    if (!fabricCanvas) return;
    zoom = Math.min(zoom * 1.2, 3);
    fabricCanvas.setZoom(zoom);
}
  function zoomOut() {
    if (!fabricCanvas) return;
    zoom = Math.max(zoom / 1.2, 0.1);
    fabricCanvas.setZoom(zoom);
}
  function resetZoom() {
    if (!fabricCanvas) return;
    zoom = 1;
    fabricCanvas.setZoom(1);
    fabricCanvas.viewportTransform = [1, 0, 0, 1, 0, 0];
}
  function deleteSelected() {
    if (!fabricCanvas || readonly) return;

    const activeObjects = fabricCanvas.getActiveObjects();
    if (activeObjects.length > 0) {
      activeObjects.forEach((obj: any) => {
        fabricCanvas.remove(obj);
      });
      fabricCanvas.discardActiveObject();
      saveCanvasState();
}}
  async function saveCanvas() {
    if (!fabricCanvas) return;

    const canvasData = JSON.stringify(
      fabricCanvas.toJSON(["evidenceId", "evidenceType", "customType"])
    );
    const positions = fabricCanvas
      .getObjects()
      .filter((obj: any) => obj.evidenceId)
      .map((obj: any) => ({
        evidenceId: obj.evidenceId,
        x: obj.left,
        y: obj.top,
        width: obj.width * obj.scaleX,
        height: obj.height * obj.scaleY,
      }));

    // Wire up to SvelteKit API endpoint
    try {
      const response = await fetch("/api/canvas/save", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          caseId,
          canvasData,
          positions,
        }),
      });
      if (!response.ok) {
        throw new Error("Failed to save canvas");
}
      notifications.add({
        type: "success",
        title: "Canvas Saved",
        message: "Evidence board saved successfully.",
      });
    } catch (error) {
      notifications.add({
        type: "error",
        title: "Save Failed",
        message: "Failed to save evidence board.",
      });
}
    onsave?.();
}
  async function exportCanvas() {
    if (!fabricCanvas) return;

    try {
      const dataURL = fabricCanvas.toDataURL({
        format: "png",
        quality: 0.9,
        multiplier: 2,
      });

      // Create download link
      const link = document.createElement("a");
      link.download = `evidence-board-${caseId || "canvas"}-${new Date().getTime()}.png`;
      link.href = dataURL;
      link.click();

      notifications.add({
        type: "success",
        title: "Export Complete",
        message: "Evidence board exported successfully.",
      });
    } catch (error) {
      console.error("Export error:", error);
      notifications.add({
        type: "error",
        title: "Export Failed",
        message: "Failed to export evidence board.",
      });
}}
  function clearCanvas() {
    if (!fabricCanvas || readonly) return;

    if (confirm("Are you sure you want to clear the entire canvas?")) {
      fabricCanvas.clear();
      fabricCanvas.backgroundColor = "#f8fafc";
      saveCanvasState();
}}
</script>

<div class="space-y-4">
  <!-- Toolbar -->
  <div
    class="space-y-4"
  >
    <div class="space-y-4">
      <!-- Tool Selection -->
      <div
        class="space-y-4"
      >
        <Button
          variant={selectedTool === "select" ? "primary" : "outline"}
          size="sm"
          onclick={() => selectTool("select")}
          disabled={readonly}
        >
          <Move class="space-y-4" />
        </Button>
        <Button
          variant={selectedTool === "draw" ? "primary" : "outline"}
          size="sm"
          onclick={() => selectTool("draw")}
          disabled={readonly}
        >
          ✏️
        </Button>
        <Button
          variant={selectedTool === "text" ? "primary" : "outline"}
          size="sm"
          onclick={() => selectTool("text")}
          disabled={readonly}
        >
          <Type class="space-y-4" />
        </Button>
      </div>

      <!-- Shapes -->
      {#if !readonly}
        <div
          class="space-y-4"
        >
          <Button
            variant="outline"
            size="sm"
            onclick={() => addShape("rectangle")}
          >
            <Square class="space-y-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onclick={() => addShape("circle")}
          >
            <Circle class="space-y-4" />
          </Button>
        </div>
      {/if}

      <!-- History -->
      <div
        class="space-y-4"
      >
        <Button
          variant="outline"
          size="sm"
          onclick={() => undo()}
          disabled={readonly || historyIndex <= 0}
        >
          <Undo class="space-y-4" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          onclick={() => redo()}
          disabled={readonly || historyIndex >= canvasHistory.length - 1}
        >
          <Redo class="space-y-4" />
        </Button>
      </div>

      <!-- Zoom -->
      <div class="space-y-4">
        <Button variant="outline" size="sm" onclick={() => zoomOut()}>
          <ZoomOut class="space-y-4" />
        </Button>
        <span class="space-y-4"
          >{Math.round(zoom * 100)}%</span
        >
        <Button variant="outline" size="sm" onclick={() => zoomIn()}>
          <ZoomIn class="space-y-4" />
        </Button>
        <Button variant="outline" size="sm" onclick={() => resetZoom()}
          >Reset</Button
        >
      </div>
    </div>

    <!-- Actions -->
    <div class="space-y-4">
      {#if !readonly}
        <Button variant="outline" size="sm" onclick={() => deleteSelected()}>
          <Trash2 class="space-y-4" />
        </Button>
        <Button variant="outline" size="sm" onclick={() => saveCanvas()}>
          <Save class="space-y-4" />
          Save
        </Button>
      {/if}
      <Button variant="outline" size="sm" onclick={() => exportCanvas()}>
        <Download class="space-y-4" />
        Export
      </Button>
    </div>
  </div>

  <!-- Canvas Container -->
  <div class="space-y-4">
    <div
      bind:this={canvasContainer}
      class="space-y-4"
    ></div>

    {#if !fabricLoaded}
      <div
        class="space-y-4"
      >
        <div class="space-y-4">
          <div
            class="space-y-4"
          ></div>
          <p class="space-y-4">Loading canvas...</p>
        </div>
      </div>
    {/if}
  </div>

  <!-- Instructions -->
  {#if fabricLoaded && evidenceItems.length === 0}
    <div
      class="space-y-4"
    >
      <Image class="space-y-4" />
      <p class="space-y-4">Evidence Board</p>
      <p class="space-y-4">
        Add evidence items to start building your case visualization
      </p>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
</style>

<!-- Enhanced Interactive Canvas with Fabric.js, No VDOM, Auto-save with Loki.js -->
<script lang="ts">
  import { aiSummarizationService } from "$lib/services/aiSummarizationService";
  import { evidenceStore } from "$lib/stores/evidenceStore";
  import type { TEvent } from "fabric";
  import * as fabric from "fabric";
  import Fuse from "fuse.js";
  import Loki from "lokijs";
  import {
    Archive,
    Circle,
    Clock,
    Copy,
    Download,
    FileText,
    Grid,
    Hand,
    Image as ImageIcon,
    MapPin,
    Minus,
    MousePointer,
    Move,
    Redo,
    Save,
    Search,
    Square,
    Trash2,
    Type,
    Undo,
    Users,
    ZoomIn,
    ZoomOut,
  } from "lucide-svelte";
  import { onDestroy, onMount } from "svelte";
  import { get, writable } from "svelte/store";

  export let caseId: string;
  export let canvasId: string = "";
  export let width = 1200;
  export let height = 800;
  export let readOnly = false;

  let canvasElement: HTMLCanvasElement;
  let canvas: fabric.Canvas | null = null;
  let lokiDb: Loki | null = null;
  let canvasCollection: Collection<any> | null = null;
  let searchEngine: Fuse<any> | null = null;

  // Canvas state management
  const canvasState = writable({
    tool: "select",
    zoom: 100,
    showGrid: true,
    showRulers: true,
    snapToGrid: true,
    gridSize: 20,
    selectedObjects: [],
    canUndo: false,
    canRedo: false,
    isDrawing: false,
    objectCount: 0,
    searchQuery: "",
    layers: [],
  });

  // History management
  let historyStack: string[] = [];
  let historyIndex = -1;
  const maxHistorySize = 50;

  // Auto-save
  let autoSaveTimeout: NodeJS.Timeout;
  let isDirty = false;

  // Tools and modes
  const tools = [
    { id: "select", icon: MousePointer, label: "Select" },
    { id: "pan", icon: Hand, label: "Pan" },
    { id: "text", icon: Type, label: "Text" },
    { id: "rect", icon: Square, label: "Rectangle" },
    { id: "circle", icon: Circle, label: "Circle" },
    { id: "line", icon: Minus, label: "Line" },
    { id: "arrow", icon: Move, label: "Arrow" },
    { id: "image", icon: ImageIcon, label: "Image" },
    { id: "evidence", icon: Archive, label: "Evidence" },
    { id: "note", icon: FileText, label: "Note" },
    { id: "timeline", icon: Clock, label: "Timeline" },
    { id: "person", icon: Users, label: "Person" },
    { id: "location", icon: MapPin, label: "Location" },
  ];

  // Evidence items from store
  let evidenceItems: unknown[] = [];
  let searchResults: unknown[] = [];

  onMount(() => {
    // Initialize components
    initializeLokiDB();
    initializeCanvas();
    initializeSearch();
    setupEventListeners();
    loadCanvasData();

    // Subscribe to evidence store
    const unsubscribe = evidenceStore.evidence.subscribe((items) => {
      evidenceItems = items;
      updateSearchEngine();
    });

    return unsubscribe;
  });

  onDestroy(() => {
    if (autoSaveTimeout) {
      clearTimeout(autoSaveTimeout);
    }
    if (canvas) {
      canvas.dispose();
    }
    if (lokiDb) {
      lokiDb.close();
    }
  });

  function initializeLokiDB() {
    lokiDb = new Loki(`canvas_${caseId}.db`, {
      autoload: true,
      autoloadCallback: () => {
        canvasCollection =
          lokiDb?.getCollection("canvas_data") ||
          lokiDb?.addCollection("canvas_data", {
            indices: ["id", "caseId", "timestamp", "type"],
          });
      },
      autosave: true,
      autosaveInterval: 4000,
    });
  }

  function initializeCanvas() {
    canvas = new fabric.Canvas(canvasElement, {
      width,
      height,
      backgroundColor: "#ffffff",
      selection: !readOnly,
      isDrawingMode: false,
      preserveObjectStacking: true,
      enableRetinaScaling: true,
    });

    // Canvas event listeners
    canvas.on("object:added", handleObjectAdded);
    canvas.on("object:removed", handleObjectRemoved);
    canvas.on("object:modified", handleObjectModified);
    canvas.on("selection:created", (options) =>
      handleSelectionCreated(options)
    );
    canvas.on("selection:updated", (options) =>
      handleSelectionUpdated(options)
    );
    canvas.on("selection:cleared", handleSelectionCleared);
    canvas.on("path:created", handlePathCreated);

    // Setup grid
    updateGrid();

    // Initialize history
    saveState();
  }

  function initializeSearch() {
    const searchOptions = {
      keys: ["title", "description", "evidenceType", "tags"],
      threshold: 0.3,
      includeMatches: true,
    };
    searchEngine = new Fuse(evidenceItems, searchOptions);
  }

  function updateSearchEngine() {
    if (searchEngine && evidenceItems) {
      searchEngine.setCollection(evidenceItems);
    }
  }

  function setupEventListeners() {
    // Keyboard shortcuts
    document.addEventListener("keydown", handleKeyboard);

    // Mouse events for tools
    if (canvas) {
      canvas.on("mouse:down", handleMouseDown);
      canvas.on("mouse:move", handleMouseMove);
      canvas.on("mouse:up", handleMouseUp);
    }
  }

  function handleKeyboard(e: KeyboardEvent) {
    if (!canvas) return;

    if (e.ctrlKey || e.metaKey) {
      switch (e.key) {
        case "z":
          e.preventDefault();
          if (e.shiftKey) {
            redo();
          } else {
            undo();
          }
          break;
        case "s":
          e.preventDefault();
          saveCanvas();
          break;
        case "c":
          e.preventDefault();
          copySelected();
          break;
        case "v":
          e.preventDefault();
          pasteClipboard();
          break;
        case "a":
          e.preventDefault();
          selectAll();
          break;
        case "Delete":
        case "Backspace":
          e.preventDefault();
          deleteSelected();
          break;
      }
    }
  }

  function handleMouseDown(event: TEvent) {
    const state = get(canvasState);
    if (!canvas) return;

    const pointer = canvas.getPointer(event.e);

    switch (state.tool) {
      case "rect":
        createRectangle(pointer);
        break;
      case "circle":
        createCircle(pointer);
        break;
      case "text":
        createText(pointer);
        break;
      case "line":
        createLine(pointer);
        break;
      case "arrow":
        createArrow(pointer);
        break;
    }
  }

  function handleMouseMove(event: TEvent) {
    // Handle drawing modes
  }

  function handleMouseUp(event: TEvent) {
    // Finalize drawing operations
  }

  function createRectangle(pointer: fabric.Point) {
    if (!canvas) return;

    const rect = new fabric.Rect({
      left: pointer.x,
      top: pointer.y,
      width: 100,
      height: 80,
      fill: "transparent",
      stroke: "#3b82f6",
      strokeWidth: 2,
      rx: 5,
      ry: 5,
    });

    canvas.add(rect);
    canvas.setActiveObject(rect);
  }

  function createCircle(pointer: fabric.Point) {
    if (!canvas) return;

    const circle = new fabric.Circle({
      left: pointer.x,
      top: pointer.y,
      radius: 50,
      fill: "transparent",
      stroke: "#10b981",
      strokeWidth: 2,
    });

    canvas.add(circle);
    canvas.setActiveObject(circle);
  }

  function createText(pointer: fabric.Point) {
    if (!canvas) return;

    const text = new fabric.IText("Click to edit text", {
      left: pointer.x,
      top: pointer.y,
      fontFamily: "Inter",
      fontSize: 16,
      fill: "#374151",
    });

    canvas.add(text);
    canvas.setActiveObject(text);
    text.enterEditing();
  }

  function createLine(pointer: fabric.Point) {
    if (!canvas) return;

    const line = new fabric.Line(
      [pointer.x, pointer.y, pointer.x + 100, pointer.y],
      {
        stroke: "#ef4444",
        strokeWidth: 2,
        selectable: true,
      }
    );

    canvas.add(line);
    canvas.setActiveObject(line);
  }

  function createArrow(pointer: fabric.Point) {
    if (!canvas) return;

    // Create arrow using a group of line and triangle
    const line = new fabric.Line([0, 0, 100, 0], {
      stroke: "#8b5cf6",
      strokeWidth: 2,
    });

    const triangle = new fabric.Triangle({
      left: 95,
      top: -5,
      width: 10,
      height: 10,
      fill: "#8b5cf6",
      angle: 90,
    });

    const arrow = new fabric.Group([line, triangle], {
      left: pointer.x,
      top: pointer.y,
    });

    canvas.add(arrow);
    canvas.setActiveObject(arrow);
  }

  function createEvidenceObject(evidence: unknown): fabric.Group {
    const rect = new fabric.Rect({
      width: 200,
      height: 150,
      fill: "#fef3c7",
      stroke: "#f59e0b",
      strokeWidth: 2,
      rx: 8,
      ry: 8,
    });

    const title = new fabric.Text(evidence.title, {
      fontSize: 14,
      fontWeight: "bold",
      top: 10,
      left: 10,
      width: 180,
    });

    const type = new fabric.Text(`Type: ${evidence.evidenceType}`, {
      fontSize: 12,
      top: 30,
      left: 10,
      fill: "#6b7280",
    });

    const description = new fabric.Text(
      evidence.description ? evidence.description.substring(0, 50) + "..." : "",
      {
        fontSize: 10,
        top: 50,
        left: 10,
        width: 180,
        fill: "#374151",
      }
    );

    // Add thumbnail if available
    let thumbnail = null;
    if (evidence.fileUrl) {
      thumbnail = createThumbnail(evidence);
    }

    const elements = [rect, title, type, description];
    if (thumbnail) elements.push(thumbnail);

    const group = new fabric.Group(elements, {
      left: 100,
      top: 100,
      hasControls: true,
      hasBorders: true,
    });

    // Store evidence data
    group.set("evidenceData", evidence);
    group.set("objectType", "evidence");

    return group;
  }

  function createThumbnail(evidence: unknown): fabric.Object | null {
    // Create appropriate thumbnail based on file type
    const fileType = evidence.fileType || evidence.mimeType || "";

    if (fileType.startsWith("image/")) {
      // For images, load and display thumbnail
      return new fabric.Rect({
        width: 60,
        height: 60,
        fill: "#e5e7eb",
        top: 80,
        left: 130,
        rx: 4,
        ry: 4,
      });
    } else if (fileType === "application/pdf") {
      // PDF icon
      return new fabric.Text("PDF", {
        fontSize: 12,
        fontWeight: "bold",
        top: 100,
        left: 140,
        fill: "#dc2626",
      });
    } else if (fileType.startsWith("video/")) {
      // Video icon
      return new fabric.Text("VIDEO", {
        fontSize: 10,
        fontWeight: "bold",
        top: 100,
        left: 140,
        fill: "#7c2d12",
      });
    }

    return null;
  }

  function addTimelineToCanvas() {
    if (!canvas) return;

    // Create a simple timeline visualization
    const timelineGroup = createTimelineVisualization();
    canvas.add(timelineGroup);
    canvas.setActiveObject(timelineGroup);
  }

  function createTimelineVisualization(): fabric.Group {
    const line = new fabric.Line([0, 0, 400, 0], {
      stroke: "#374151",
      strokeWidth: 3,
    });

    const elements: fabric.FabricObject[] = [line];

    // Add timeline markers
    for (let i = 0; i <= 4; i++) {
      const marker = new fabric.Circle({
        left: i * 100 - 5,
        top: -5,
        radius: 5,
        fill: "#3b82f6",
        stroke: "#1e40af",
        strokeWidth: 1,
      });

      const date = new fabric.Text(`Event ${i + 1}`, {
        left: i * 100 - 25,
        top: 15,
        fontSize: 10,
        fill: "#374151",
      });

      elements.push(marker, date);
    }

    const timeline = new fabric.Group(elements, {
      left: 100,
      top: 200,
    });

    timeline.set("objectType", "timeline");
    return timeline;
  }

  function addPersonToCanvas() {
    if (!canvas) return;

    const person = createPersonVisualization();
    canvas.add(person);
    canvas.setActiveObject(person);
  }

  function createPersonVisualization(): fabric.Group {
    const circle = new fabric.Circle({
      radius: 30,
      fill: "#dbeafe",
      stroke: "#3b82f6",
      strokeWidth: 2,
    });

    const name = new fabric.Text("Person Name", {
      fontSize: 12,
      fontWeight: "bold",
      top: 40,
      left: -30,
      textAlign: "center",
    });

    const role = new fabric.Text("Role/Title", {
      fontSize: 10,
      top: 55,
      left: -25,
      fill: "#6b7280",
      textAlign: "center",
    });

    const person = new fabric.Group([circle, name, role], {
      left: 200,
      top: 200,
    });

    person.set("objectType", "person");
    return person;
  }

  function addLocationToCanvas() {
    if (!canvas) return;

    const location = createLocationVisualization();
    canvas.add(location);
    canvas.setActiveObject(location);
  }

  function createLocationVisualization(): fabric.Group {
    const marker = new fabric.Polygon(
      [
        { x: 0, y: 0 },
        { x: 20, y: 0 },
        { x: 30, y: 15 },
        { x: 20, y: 30 },
        { x: 0, y: 30 },
        { x: 10, y: 15 },
      ],
      {
        fill: "#dc2626",
        stroke: "#991b1b",
        strokeWidth: 1,
      }
    );

    const label = new fabric.Text("Location", {
      fontSize: 12,
      top: 35,
      left: -10,
      fill: "#374151",
    });

    const location = new fabric.Group([marker, label], {
      left: 300,
      top: 200,
    });

    location.set("objectType", "location");
    return location;
  }

  // Canvas state management functions
  function handleObjectAdded() {
    isDirty = true;
    updateCanvasState();
    scheduleAutoSave();
    saveState();
  }

  function handleObjectRemoved() {
    isDirty = true;
    updateCanvasState();
    scheduleAutoSave();
    saveState();
  }

  function handleObjectModified() {
    isDirty = true;
    scheduleAutoSave();
    saveState();
  }

  function handleSelectionCreated(options: unknown) {
    updateSelection();
  }

  function handleSelectionUpdated(options: unknown) {
    updateSelection();
  }

  function handleSelectionCleared() {
    updateSelection();
  }

  function handlePathCreated() {
    saveState();
  }

  function updateSelection() {
    if (!canvas) return;

    const activeObjects = canvas.getActiveObjects();
    canvasState.update((state) => ({
      ...state,
      selectedObjects: activeObjects,
    }));
  }

  function updateCanvasState() {
    if (!canvas) return;

    canvasState.update((state) => ({
      ...state,
      objectCount: canvas.getObjects().length,
      canUndo: historyIndex > 0,
      canRedo: historyIndex < historyStack.length - 1,
    }));
  }

  // History management
  function saveState() {
    if (!canvas) return;

    const state = JSON.stringify(canvas.toJSON());

    // Remove states after current index
    historyStack = historyStack.slice(0, historyIndex + 1);

    // Add new state
    historyStack.push(state);
    historyIndex = historyStack.length - 1;

    // Limit history size
    if (historyStack.length > maxHistorySize) {
      historyStack.shift();
      historyIndex--;
    }

    updateCanvasState();
  }

  async function undo() {
    if (!canvas || historyIndex <= 0) return;

    historyIndex--;
    const state = historyStack[historyIndex];
    await canvas.loadFromJSON(state);
    canvas.renderAll();
    updateCanvasState();
  }

  async function redo() {
    if (!canvas || historyIndex >= historyStack.length - 1) return;

    historyIndex++;
    const state = historyStack[historyIndex];
    await canvas.loadFromJSON(state);
    canvas.renderAll();
    updateCanvasState();
  }

  // Auto-save functionality
  function scheduleAutoSave() {
    if (autoSaveTimeout) {
      clearTimeout(autoSaveTimeout);
    }
    autoSaveTimeout = setTimeout(() => {
      saveCanvas();
    }, 3000);
  }

  async function saveCanvas() {
    if (!canvas || !canvasCollection || !isDirty) return;

    try {
      const canvasData = {
        id: canvasId || crypto.randomUUID(),
        caseId,
        data: canvas.toJSON(),
        thumbnail: canvas.toDataURL({ format: "png", multiplier: 0.1 }),
        metadata: {
          objectCount: canvas.getObjects().length,
          width: canvas.getWidth(),
          height: canvas.getHeight(),
          zoom: get(canvasState).zoom,
        },
        timestamp: new Date(),
        version: Date.now(),
      };

      // Save to Loki.js
      const existing = canvasCollection.findOne({ id: canvasData.id });
      if (existing) {
        canvasCollection.update({ ...existing, ...canvasData });
      } else {
        canvasCollection.insert(canvasData);
      }

      // Save to server
      await fetch("/api/canvas/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(canvasData),
      });

      isDirty = false;
      showSaveIndicator();
    } catch (error) {
      console.error("Failed to save canvas:", error);
    }
  }

  async function loadCanvasData() {
    if (!canvasCollection) return;

    try {
      // Try loading from Loki.js first
      const localData = canvasCollection.findOne({ caseId });

      if (localData) {
        canvas?.loadFromJSON(localData.data, () => {
          canvas?.renderAll();
        });
        return;
      }

      // Fallback to server
      const response = await fetch(`/api/canvas/${caseId}`);
      if (response.ok) {
        const serverData = await response.json();
        canvas?.loadFromJSON(serverData.data, () => {
          canvas?.renderAll();
        });
      }
    } catch (error) {
      console.error("Failed to load canvas:", error);
    }
  }

  function showSaveIndicator() {
    // Visual save indicator
    const indicator = document.createElement("div");
    indicator.textContent = "Canvas Saved";
    indicator.className =
      "fixed top-4 right-4 bg-green-500 text-white px-3 py-1 rounded text-sm z-50";
    document.body.appendChild(indicator);
    setTimeout(() => {
      if (document.body.contains(indicator)) {
        document.body.removeChild(indicator);
      }
    }, 2000);
  }

  // Tool functions
  function setTool(toolId: string) {
    canvasState.update((state) => ({ ...state, tool: toolId }));

    if (!canvas) return;

    // Reset canvas modes
    canvas.isDrawingMode = false;
    canvas.selection = toolId === "select";

    // Set appropriate cursor
    switch (toolId) {
      case "pan":
        canvas.defaultCursor = "grab";
        break;
      case "text":
        canvas.defaultCursor = "text";
        break;
      default:
        canvas.defaultCursor = "default";
    }
  }

  function zoomIn() {
    const currentZoom = get(canvasState).zoom;
    const newZoom = Math.min(currentZoom + 10, 200);
    setZoom(newZoom);
  }

  function zoomOut() {
    const currentZoom = get(canvasState).zoom;
    const newZoom = Math.max(currentZoom - 10, 25);
    setZoom(newZoom);
  }

  function setZoom(zoom: number) {
    if (!canvas) return;

    canvasState.update((state) => ({ ...state, zoom }));
    canvas.setZoom(zoom / 100);
    canvas.renderAll();
  }

  function toggleGrid() {
    canvasState.update((state) => ({ ...state, showGrid: !state.showGrid }));
    updateGrid();
  }

  function updateGrid() {
    if (!canvas) return;

    const state = get(canvasState);
    if (state.showGrid) {
      // Add grid pattern
      const gridSize = state.gridSize;
      const pattern = createGridPattern(gridSize);
      canvas.backgroundColor = "#f8f9fa"; // Light gray background for grid
      canvas.renderAll();
    } else {
      canvas.backgroundColor = "#ffffff";
      canvas.renderAll();
    }
  }

  function createGridPattern(size: number): string {
    const svg = `
      <svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="grid" width="${size}" height="${size}" patternUnits="userSpaceOnUse">
            <path d="M ${size} 0 L 0 0 0 ${size}" fill="none" stroke="#e5e7eb" stroke-width="1"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
      </svg>
    `;
    return `data:image/svg+xml;base64,${btoa(svg)}`;
  }

  // Object manipulation
  function deleteSelected() {
    if (!canvas) return;

    const activeObjects = canvas.getActiveObjects();
    if (activeObjects.length) {
      canvas.discardActiveObject();
      activeObjects.forEach((obj) => canvas?.remove(obj));
      canvas.renderAll();
    }
  }

  async function copySelected() {
    if (!canvas) return;

    const activeObject = canvas.getActiveObject();
    if (activeObject) {
      const cloned = await activeObject.clone();
      cloned.set({
        left: (cloned.left || 0) + 10,
        top: (cloned.top || 0) + 10,
      });
      canvas?.add(cloned);
      canvas?.setActiveObject(cloned);
      canvas?.renderAll();
    }
  }

  function pasteClipboard() {
    // Implement clipboard paste functionality
    navigator.clipboard.read().then((items) => {
      for (const item of items) {
        if (item.types.includes("image/png")) {
          item.getType("image/png").then((blob) => {
            const reader = new FileReader();
            reader.onload = async (e) => {
              const imgUrl = e.target?.result as string;
              const img = await fabric.Image.fromURL(imgUrl);
              img.scale(0.5);
              canvas?.add(img);
              canvas?.renderAll();
            };
            reader.readAsDataURL(blob);
          });
        }
      }
    });
  }

  function selectAll() {
    if (!canvas) return;

    const allObjects = canvas.getObjects();
    const selection = new fabric.ActiveSelection(allObjects, { canvas });
    canvas.setActiveObject(selection);
    canvas.renderAll();
  }

  // Export functions
  function exportCanvas(format: "png" | "svg" | "json") {
    if (!canvas) return;

    let dataUrl: string;
    let filename: string;

    switch (format) {
      case "png":
        dataUrl = canvas.toDataURL({
          format: "png",
          quality: 1,
          multiplier: 1,
        });
        filename = `canvas-${caseId}.png`;
        break;
      case "svg":
        dataUrl = `data:image/svg+xml;base64,${btoa(canvas.toSVG())}`;
        filename = `canvas-${caseId}.svg`;
        break;
      case "json":
        const jsonData = canvas.toJSON();
        dataUrl = `data:application/json;base64,${btoa(JSON.stringify(jsonData, null, 2))}`;
        filename = `canvas-${caseId}.json`;
        break;
    }

    const link = document.createElement("a");
    link.href = dataUrl;
    link.download = filename;
    link.click();
  }

  // Search functionality
  function searchEvidence(query: string) {
    if (!searchEngine || !query.trim()) {
      searchResults = [];
      return;
    }

    const results = searchEngine.search(query);
    searchResults = results.map((result) => result.item);
  }

  // Generate AI summary for canvas
  async function generateAISummary() {
    if (!canvas) return;

    try {
      const canvasObjects = canvas.getObjects();
      const evidenceObjects = canvasObjects.filter(
        (obj) => obj.get("objectType") === "evidence"
      );
      const evidenceData = evidenceObjects
        .map((obj) => obj.get("evidenceData"))
        .filter(Boolean);

      if (evidenceData.length === 0) {
        alert("No evidence items found on canvas to summarize.");
        return;
      }

      const summary = await aiSummarizationService.generateEvidenceAnalysis(
        evidenceData,
        caseId
      );

      // Add summary as a text object on canvas
      const summaryText = new fabric.IText(
        `AI SUMMARY:\n${summary.content.substring(0, 200)}...`,
        {
          left: 50,
          top: 50,
          width: 300,
          fontSize: 12,
          fill: "#374151",
          backgroundColor: "#f0f9ff",
          padding: 10,
        }
      );

      canvas.add(summaryText);
      canvas.renderAll();
    } catch (error) {
      console.error("Failed to generate AI summary:", error);
      alert("Failed to generate AI summary. Please try again.");
    }
  }

  // Reactive statements
  $: state = get(canvasState);

  // Exported functions for parent component access
  export function addEvidenceToCanvas(evidence: unknown) {
    if (!canvas) return;
    const evidenceObject = createEvidenceObject(evidence);
    canvas.add(evidenceObject);
    canvas.setActiveObject(evidenceObject);
  }

  export function addElementsToCanvas(elements: unknown[]) {
    if (!canvas || !elements) return;
    elements.forEach((element) => {
      // Create canvas objects from element data
      const canvasObject = createCanvasObjectFromData(element);
      if (canvasObject) {
        canvas.add(canvasObject);
      }
    });
    canvas.renderAll();
  }

  function createCanvasObjectFromData(elementData: unknown): fabric.Object | null {
    try {
      // Basic implementation - can be expanded based on element types
      if (elementData.type === "evidence") {
        return createEvidenceObject(elementData);
      } else if (elementData.type === "text") {
        return new fabric.Text(elementData.text || "Text", {
          left: elementData.left || 100,
          top: elementData.top || 100,
          fontSize: elementData.fontSize || 16,
          fill: elementData.fill || "#333",
        });
      }
      return null;
    } catch (error) {
      console.error("Error creating canvas object:", error);
      return null;
    }
  }
</script>

<div class="mx-auto px-4 max-w-7xl">
  <!-- Main Toolbar -->
  <div
    class="mx-auto px-4 max-w-7xl"
  >
    <!-- File Operations -->
    <div class="mx-auto px-4 max-w-7xl">
      <button
        class="mx-auto px-4 max-w-7xl"
        onclick={() => saveCanvas()}
        title="Save Canvas"
      >
        <Save size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        onclick={() => undo()}
        disabled={!state.canUndo}
        title="Undo"
      >
        <Undo size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        onclick={() => redo()}
        disabled={!state.canRedo}
        title="Redo"
      >
        <Redo size="18" />
      </button>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- Tools -->
    <div class="mx-auto px-4 max-w-7xl">
      {#each tools as tool}
        <button
          class="mx-auto px-4 max-w-7xl"
          class:active={state.tool === tool.id}
          onclick={() => setTool(tool.id)}
          title={tool.label}
        >
          <svelte:component this={tool.icon} size="18" />
        </button>
      {/each}
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- Canvas Controls -->
    <div class="mx-auto px-4 max-w-7xl">
      <button class="mx-auto px-4 max-w-7xl" onclick={() => zoomOut()} title="Zoom Out">
        <ZoomOut size="18" />
      </button>
      <span class="mx-auto px-4 max-w-7xl">{state.zoom}%</span>
      <button class="mx-auto px-4 max-w-7xl" onclick={() => zoomIn()} title="Zoom In">
        <ZoomIn size="18" />
      </button>

      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.showGrid}
        onclick={() => toggleGrid()}
        title="Toggle Grid"
      >
        <Grid size="18" />
      </button>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- Object Actions -->
    <div class="mx-auto px-4 max-w-7xl">
      <button class="mx-auto px-4 max-w-7xl" onclick={() => copySelected()} title="Copy">
        <Copy size="18" />
      </button>
      <button class="mx-auto px-4 max-w-7xl" onclick={() => pasteClipboard()} title="Paste">
        <Copy size="18" />
      </button>
      <button class="mx-auto px-4 max-w-7xl" onclick={() => deleteSelected()} title="Delete">
        <Trash2 size="18" />
      </button>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- AI Features -->
    <div class="mx-auto px-4 max-w-7xl">
      <button
        class="mx-auto px-4 max-w-7xl"
        onclick={() => generateAISummary()}
        title="Generate AI Summary"
      >
        <FileText size="18" />
      </button>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- Export -->
    <div class="mx-auto px-4 max-w-7xl">
      <button class="mx-auto px-4 max-w-7xl">
        <Download size="18" />
      </button>
      <div class="mx-auto px-4 max-w-7xl">
        <button onclick={() => exportCanvas("png")}>Export as PNG</button>
        <button onclick={() => exportCanvas("svg")}>Export as SVG</button>
        <button onclick={() => exportCanvas("json")}>Export as JSON</button>
      </div>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- Canvas Info -->
    <div class="mx-auto px-4 max-w-7xl">
      Objects: {state.objectCount} | Selected: {state.selectedObjects.length}
    </div>
  </div>

  <!-- Content Area -->
  <div class="mx-auto px-4 max-w-7xl">
    <!-- Evidence Sidebar -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <Search size="16" />
          <input
            type="text"
            placeholder="Search evidence..."
            bind:value={state.searchQuery}
            oninput={(e) =>
              searchEvidence((e.target as HTMLInputElement).value)}
            class="mx-auto px-4 max-w-7xl"
          />
        </div>

        <h3 class="mx-auto px-4 max-w-7xl">Evidence Items</h3>
        <div class="mx-auto px-4 max-w-7xl">
          {#each state.searchQuery ? searchResults : evidenceItems as evidence}
            <div
              class="mx-auto px-4 max-w-7xl"
              onclick={() => addEvidenceToCanvas(evidence)}
              onkeydown={(e) =>
                e.key === "Enter" && addEvidenceToCanvas(evidence)}
              role="button"
              tabindex={0}
            >
              <div class="mx-auto px-4 max-w-7xl">{evidence.title}</div>
              <div class="mx-auto px-4 max-w-7xl">{evidence.evidenceType}</div>
            </div>
          {/each}
        </div>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <h3 class="mx-auto px-4 max-w-7xl">Quick Add</h3>
        <div class="mx-auto px-4 max-w-7xl">
          <button
            class="mx-auto px-4 max-w-7xl"
            onclick={() => addTimelineToCanvas()}
          >
            <Clock size="16" class="mx-auto px-4 max-w-7xl" />
            Timeline
          </button>
          <button
            class="mx-auto px-4 max-w-7xl"
            onclick={() => addPersonToCanvas()}
          >
            <Users size="16" class="mx-auto px-4 max-w-7xl" />
            Person
          </button>
          <button
            class="mx-auto px-4 max-w-7xl"
            onclick={() => addLocationToCanvas()}
          >
            <MapPin size="16" class="mx-auto px-4 max-w-7xl" />
            Location
          </button>
          <button
            class="mx-auto px-4 max-w-7xl"
            onclick={() => setTool("note")}
          >
            <FileText size="16" class="mx-auto px-4 max-w-7xl" />
            Note
          </button>
        </div>
      </div>
    </div>

    <!-- Canvas Area -->
    <div class="mx-auto px-4 max-w-7xl">
      <canvas bind:this={canvasElement} class="mx-auto px-4 max-w-7xl"></canvas>
    </div>
  </div>
</div>

<style>
  .canvas-editor-container {
    background: #f9fafb;
  }

  .toolbar {
    min-height: 48px;
  }

  .toolbar-group {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }

  .toolbar-btn {
    padding: 0.5rem;
    border-radius: 0.375rem;
    transition: background-color 0.2s;
    border: none;
    background: transparent;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 36px;
    height: 36px;
  }

  .toolbar-btn:hover {
    background-color: #f3f4f6;
  }

  .toolbar-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .toolbar-btn.active {
    background-color: #dbeafe;
    color: #2563eb;
  }

  .toolbar-separator {
    width: 1px;
    height: 1.5rem;
    background-color: #d1d5db;
    margin: 0 0.25rem;
  }

  .zoom-display {
    font-size: 0.875rem;
    color: #4b5563;
    min-width: 40px;
    text-align: center;
  }

  .dropdown {
    position: relative;
  }

  .dropdown-menu {
    position: absolute;
    top: 100%;
    left: 0;
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 0.375rem;
    box-shadow:
      0 10px 15px -3px rgba(0, 0, 0, 0.1),
      0 4px 6px -2px rgba(0, 0, 0, 0.05);
    padding: 0.25rem 0;
    z-index: 20;
    min-width: 150px;
    display: none;
  }

  .dropdown:hover .dropdown-menu {
    display: block;
  }

  .dropdown-menu button {
    width: 100%;
    text-align: left;
    padding: 0.5rem 0.75rem;
    border: none;
    background-color: transparent;
    cursor: pointer;
  }

  .dropdown-menu button:hover {
    background-color: #f3f4f6;
  }

  .evidence-item {
    transition: all 0.2s ease;
  }

  .evidence-item:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .canvas-area {
    background:
      linear-gradient(to right, #e5e7eb 1px, transparent 1px),
      linear-gradient(to bottom, #e5e7eb 1px, transparent 1px);
    background-size: 20px 20px;
  }
</style>
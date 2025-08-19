<script lang="ts">
  import { page } from "$app/stores";
  import { caseService } from "$lib/services/caseService";
  import { onMount } from "svelte";
  // Canvas Components
  import EvidenceNode from "$lib/components/canvas/EvidenceNode.svelte";
  import POINode from "$lib/components/canvas/POINode.svelte";
  import ReportNode from "$lib/components/canvas/ReportNode.svelte";
  // UI Components
  import AISummaryModal from "$lib/components/modals/AISummaryModal.svelte";
  import * as ContextMenu from "$lib/components/ui/context-menu";
  // FIX: Corrected the Button import path
  import { Button } from "$lib/components/ui/button";
  // Icons
  import { FileText, Image, Upload, User as UserIcon } from "lucide-svelte";

  // NOTE: For a large app, these types should be moved to a shared file (e.g., src/lib/types.ts)
  interface Report {
    id: string;
    title: string;
    posX: number;
    posY: number;
    caseId: string;
}
  interface Evidence {
    id: string;
    title?: string;
    name?: string;
    fileUrl?: string;
    url?: string;
    x?: number;
    y?: number;
    posX?: number;
    posY?: number;
    width?: number;
    height?: number;
    isSelected?: boolean;
    isDirty?: boolean;
}
  interface POI {
    id: string;
    name: string;
    posX: number;
    posY: number;
    relationship?: string;
    caseId: string;
}
  interface ContextMenuState {
    show: boolean;
    x: number;
    y: number;
}
  // FIX: Simplified store access
  const {
    reports,
    evidence,
    pois,
    isLoading,
    error: caseServiceError,
  } = caseService;

  $: caseId = $page.params.id;
  let canvasElement: HTMLElement;
  let contextMenuState: ContextMenuState = { show: false, x: 0, y: 0 };

  // State for uploads
  let isUploading = false;
  let uploadProgress = 0; // Note: fetch doesn't easily support progress. See function comments.

  onMount(() => {
    if (caseId) {
      caseService.loadCase(caseId);
}
  });

  function getCanvasCoordinates(clientX: number, clientY: number) {
    const rect = canvasElement.getBoundingClientRect();
    return { x: clientX - rect.left, y: clientY - rect.top };
}
  function createNewReport(x: number, y: number) {
    const coords = getCanvasCoordinates(x, y);
    caseService.createReport({
      title: "New Report",
      posX: coords.x,
      posY: coords.y,
      caseId: caseId,
    });
    contextMenuState.show = false;
}
  function createNewPOI(x: number, y: number, relationship?: string) {
    const coords = getCanvasCoordinates(x, y);
    caseService.createPOI({
      name: relationship ? `New ${relationship}` : "New Person of Interest",
      posX: coords.x,
      posY: coords.y,
      relationship: relationship,
      caseId: caseId,
    });
    contextMenuState.show = false;
}
  function createNewEvidence(x: number, y: number) {
    const input = document.createElement("input");
    input.type = "file";
    input.onchange=async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const coords = getCanvasCoordinates(x, y);
        await uploadAndCreateEvidence(file, coords.x, coords.y);
}
    };
    input.click();
    contextMenuState.show = false;
}
  // Modernized file upload using fetch
  async function uploadAndCreateEvidence(file: File, x: number, y: number) {
    if (!file || !caseId) return;
    isUploading = true;
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("caseId", caseId);

      const response = await fetch("/api/evidence/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorResult = await response.json();
        throw new Error(
          errorResult.message || `Upload failed: ${response.statusText}`
        );
}
      const result = await response.json();

      await caseService.createEvidence({
        title: file.name,
        fileUrl: result.url,
        posX: x,
        posY: y,
        caseId: caseId,
        metadata: {},
      });
    } catch (err) {
      console.error("Error uploading evidence:", err);
    } finally {
      isUploading = false;
}
}
  async function handleCanvasDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    const files = event.dataTransfer?.files;
    if (!files || files.length === 0) return;
    const file = files[0];
    const coords = getCanvasCoordinates(event.clientX, event.clientY);
    await uploadAndCreateEvidence(file, coords.x, coords.y);
}
  function handleCanvasDragOver(event: DragEvent) {
    event.preventDefault();
    event.dataTransfer!.dropEffect = "copy";
}
  function handleCanvasDragEnter(event: DragEvent) {
    event.preventDefault();
    canvasElement?.classList.add("drag-over");
}
  function handleCanvasDragLeave(event: DragEvent) {
    event.preventDefault();
    if (event.target === canvasElement) {
      canvasElement?.classList.remove("drag-over");
}
}
  function handleCanvasContextMenu(event: MouseEvent) {
    event.preventDefault();
    contextMenuState = {
      show: true,
      x: event.clientX,
      y: event.clientY,
    };
}
</script>

<svelte:head>
  <title>Case {caseId} - Interactive Canvas</title>
</svelte:head>

<div class="container mx-auto px-4">
  <!-- Canvas Area -->
  <ContextMenu.Root>
    <ContextMenu.Trigger asChild>
      <div
        bind:this={canvasElement}
        class="container mx-auto px-4"
        role="application"
        aria-label="Case evidence canvas"
        on:contextmenu={handleCanvasContextMenu}
        ondrop={handleCanvasDrop}
        ondragover={handleCanvasDragOver}
        ondragenter={handleCanvasDragEnter}
        ondragleave={handleCanvasDragLeave}
      >
        <!-- Grid Background -->
        <div class="container mx-auto px-4" aria-hidden="true"></div>

        <!-- Drop Zone Indicator -->
        {#if isUploading}
          <div class="container mx-auto px-4">
            <div class="container mx-auto px-4">
              <Upload class="container mx-auto px-4" />
              <p>Uploading... {Math.round(uploadProgress)}%</p>
              <div class="container mx-auto px-4">
                <div
                  class="container mx-auto px-4"
                  style="width: {uploadProgress}%"
                ></div>
              </div>
            </div>
          </div>
        {/if}

        <!-- Render Reports -->
        {#each $reports as report (report.id)}
          <ReportNode
            report={{
              ...report,
              status: report.status ?? "",
              createdAt:
                typeof report.createdAt === "string"
                  ? new Date(report.createdAt)
                  : (report.createdAt ?? new Date()),
              updatedAt:
                typeof report.updatedAt === "string"
                  ? new Date(report.updatedAt)
                  : (report.updatedAt ?? new Date()),
              tags: report.tags ?? [],
              content: report.content ?? "",
              isPublic: report.isPublic ?? false,
              createdBy: report.createdBy ?? "",
              metadata: report.metadata ?? {},
              caseId: report.caseId ?? caseId,
              reportType: report.reportType ?? ""
            }}
          />
        {/each}

        <!-- Render Evidence -->
        {#each $evidence as evidenceItem (evidenceItem.id)}
          <EvidenceNode
            title={evidenceItem.title || evidenceItem.name || "Evidence"}
            fileUrl={evidenceItem.fileUrl || evidenceItem.url || ""}
            position={{
              x: evidenceItem.posX ?? evidenceItem.x ?? 0,
              y: evidenceItem.posY ?? evidenceItem.y ?? 0,
            }}
            size={{
              width: evidenceItem.width || 200,
              height: evidenceItem.height || 150,
            }}
            isSelected={evidenceItem.isSelected || false}
            isDirty={evidenceItem.isDirty || false}
          />
        {/each}

        <!-- Render POIs -->
        {#each $pois as poi (poi.id)}
          <POINode {poi} />
        {/each}

        <!-- Loading overlay -->
        {#if $isLoading}
          <div class="container mx-auto px-4" aria-live="polite">
            <div class="container mx-auto px-4" aria-hidden="true"></div>
            <p>Loading case data...</p>
          </div>
        {/if}
      </div>
    </ContextMenu.Trigger>

    <ContextMenu.Content class="container mx-auto px-4">
      <ContextMenu.Item
        on:click={() => createNewReport(contextMenuState.x, contextMenuState.y)}
      >
        <FileText class="container mx-auto px-4" />
        New Report
      </ContextMenu.Item>
      <ContextMenu.Item
        on:click={() =>
          createNewEvidence(contextMenuState.x, contextMenuState.y)}
      >
        <Image class="container mx-auto px-4" />
        New Evidence
      </ContextMenu.Item>
      <ContextMenu.Separator />
      <ContextMenu.Item
        on:click={() =>
          createNewPOI(contextMenuState.x, contextMenuState.y, "suspect")}
      >
        <UserIcon class="container mx-auto px-4" />
        Add Suspect
      </ContextMenu.Item>
      <ContextMenu.Item
        on:click={() =>
          createNewPOI(contextMenuState.x, contextMenuState.y, "witness")}
      >
        <UserIcon class="container mx-auto px-4" />
        Add Witness
      </ContextMenu.Item>
      <ContextMenu.Item
        on:click={() =>
          createNewPOI(contextMenuState.x, contextMenuState.y, "victim")}
      >
        <UserIcon class="container mx-auto px-4" />
        Add Victim
      </ContextMenu.Item>
      <ContextMenu.Item
        on:click={() =>
          createNewPOI(
            contextMenuState.x,
            contextMenuState.y,
            "co-conspirator"
          )}
      >
        <UserIcon class="container mx-auto px-4" />
        Add Co-conspirator
      </ContextMenu.Item>
      <ContextMenu.Item
        on:click={() =>
          createNewPOI(contextMenuState.x, contextMenuState.y, "informant")}
      >
        <UserIcon class="container mx-auto px-4" />
        Add Informant
      </ContextMenu.Item>
      <ContextMenu.Item
        on:click={() => createNewPOI(contextMenuState.x, contextMenuState.y)}
      >
        <UserIcon class="container mx-auto px-4" />
        Add Other POI
      </ContextMenu.Item>
    </ContextMenu.Content>
  </ContextMenu.Root>

  <!-- Toolbar -->
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <Button variant="secondary" on:click={() => createNewReport(100, 100)}>
        <FileText class="container mx-auto px-4" />
        New Report
      </Button>

      <Button variant="secondary" on:click={() => createNewEvidence(200, 100)}>
        <Image class="container mx-auto px-4" />
        New Evidence
      </Button>

      <Button variant="secondary" on:click={() => createNewPOI(300, 100)}>
        <UserIcon class="container mx-auto px-4" />
        New POI
      </Button>
    </div>

    <div class="container mx-auto px-4">
      <Button variant="secondary" on:click={() => caseService.saveAll()}>
        Save All
      </Button>
    </div>
  </div>

  <!-- Status Bar -->
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <span>Reports: {$reports.length}</span>
      <span>Evidence: {$evidence.length}</span>
      <span>POIs: {$pois.length}</span>
    </div>

    {#if $caseServiceError}
      <div class="container mx-auto px-4" role="alert">
        Error: {$caseServiceError}
      </div>
    {/if}
  </div>

  <!-- Drop Zone Help -->
  <div class="container mx-auto px-4">
    <p>Drag and drop files here to add evidence</p>
  </div>
</div>

<!-- Global AI Summary Modal -->
<AISummaryModal />

<style>
  /* @unocss-include */
.canvas-background {
    position: relative;
    width: 100%;
    height: 100%;
    background: #f8fafc;
    cursor: grab;
    overflow: hidden;
    transition: background-color 0.2s;
}
  .canvas-background:active {
    cursor: grabbing;
}
  .canvas-background.drag-over {
    background-color: #e0f2fe;
    border: 2px dashed #0ea5e9;
}
  .canvas-grid {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image:
      linear-gradient(rgba(0, 0, 0, 0.1) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0, 0, 0, 0.1) 1px, transparent 1px);
    background-size: 20px 20px;
    pointer-events: none;
    opacity: 0.3;
}
  .upload-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.95);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 150;
}
  .upload-progress {
    text-align: center;
    color: #0ea5e9;
}
  .progress-bar {
    width: 200px;
    height: 4px;
    background: #e2e8f0;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 1rem;
}
  .progress-fill {
    height: 100%;
    background: #0ea5e9;
    transition: width 0.3s ease;
}
  .canvas-toolbar {
    position: absolute;
    top: 1rem;
    left: 1rem;
    display: flex;
    gap: 1rem;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 0.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    z-index: 100;
}
  .toolbar-section {
    display: flex;
    gap: 0.5rem;
    align-items: center;
}
  .status-bar {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    border-top: 1px solid #e2e8f0;
    padding: 0.5rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 100;
    font-size: 0.875rem;
}
  .status-info {
    display: flex;
    gap: 1rem;
    color: #64748b;
}
  .status-error {
    color: #ef4444;
    font-weight: 500;
}
  .loading-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.8);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 200;
}
.drop-zone-help {
    position: absolute;
    bottom: 3rem;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.7);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
}
  /* Removed unused selector: .canvas-background.drag-over .drop-zone-help */

  @keyframes spin {
    0% {
      transform: rotate(0deg);
}
    100% {
      transform: rotate(360deg);
}
}
</style>

<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { dndzone } from "svelte-dnd-action";
  import { writable } from "svelte/store";
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/Card";
  import Button from "$lib/components/ui/button/Button.svelte";
  import Badge from "$lib/components/ui/Badge.svelte";
  import EvidenceNode from "../canvas/EvidenceNode.svelte";
  import ContextMenu from "./ContextMenu.svelte";
  import EvidenceCard from "./EvidenceCard.svelte";
  import UploadZone from "./UploadZone.svelte";

  // Define proper types
  interface EvidenceType {
    id: string;
    title: string;
    status: string;
    evidenceType: string;
    tags: string[];
    uploadedAt: Date;
    createdAt: Date;
    updatedAt: Date;
    fileUrl?: string;
    description?: string;
  }

  interface EvidenceWithPosition extends EvidenceType {
    position: { x: number; y: number };
  }

  export let caseId: string;
  export let evidence: EvidenceType[] = [];

  // View modes: 'columns' | 'canvas'
  let viewMode: "columns" | "canvas" = "columns";

  // Store for real-time updates
  const evidenceStore = writable(
    evidence.map((item) => ({
      ...item,
      status: item.status || "new",
      evidenceType: item.evidenceType || "document",
      tags: item.tags || [],
      uploadedAt: item.uploadedAt || item.createdAt || new Date(),
      createdAt: item.createdAt || new Date(),
      updatedAt: item.updatedAt || new Date(),
      position: { x: Math.random() * 400 + 100, y: Math.random() * 300 + 100 },
    })) as EvidenceWithPosition[]
  );

  // Active users in real-time collaboration
  const activeUsers = writable<Array<{ id: string; name?: string; email?: string }>>([]);

  // Canvas view state
  let canvasContainer: HTMLElement;
  let canvasEvidence: EvidenceWithPosition[] = [];

  // Column layout
  let columns: Array<{ id: string; title: string; items: EvidenceType[] }> = [
    { id: "new", title: "New Evidence", items: [] },
    { id: "reviewing", title: "Under Review", items: [] },
    { id: "approved", title: "Case Ready", items: [] },
  ];

  // Context menu state
  let contextMenu: {
    show: boolean;
    x: number;
    y: number;
    item: EvidenceType | null;
  } = {
    show: false,
    x: 0,
    y: 0,
    item: null,
  };

  // WebSocket for real-time updates
  let ws: WebSocket | null = null;

  // Subscribe to evidence changes
  $: {
    if (viewMode === "canvas") {
      canvasEvidence = $evidenceStore;
    } else {
      distributeEvidence();
    }
  }

  // WebSocket connection logic
  function connectWebSocket() {
    try {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(
        `${protocol}//${window.location.host}/ws/cases/${caseId}`
      );
      
      ws.onopen = () => {
        console.log("Connected to real-time case updates");
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleRealtimeUpdate(data);
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };
      
      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
      };
      
      ws.onclose = () => {
        console.log("Disconnected from real-time updates");
        setTimeout(() => {
          if (!ws || ws.readyState === WebSocket.CLOSED) {
            connectWebSocket();
          }
        }, 3000);
      };
    } catch (error) {
      console.warn(
        "WebSocket not available, real-time features disabled:",
        error
      );
    }
  }

  onMount(() => {
    connectWebSocket();
    distributeEvidence();
  });

  onDestroy(() => {
    if (ws) {
      ws.close();
    }
  });

  function distributeEvidence() {
    const items = $evidenceStore;
    columns = [
      {
        id: "new",
        title: "New Evidence",
        items: items.filter((item) => item.status === "new"),
      },
      {
        id: "reviewing",
        title: "Under Review",
        items: items.filter((item) => item.status === "reviewing"),
      },
      {
        id: "approved",
        title: "Case Ready",
        items: items.filter((item) => item.status === "approved"),
      },
    ];
  }

  function handleRightClick(event: MouseEvent, item: EvidenceType) {
    event.preventDefault();
    contextMenu = {
      show: true,
      x: event.clientX,
      y: event.clientY,
      item,
    };
  }

  function closeContextMenu() {
    contextMenu.show = false;
  }

  async function sendToCase(evidenceId: string, targetCaseId: string) {
    try {
      const response = await fetch("/api/evidence/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ evidenceId, caseId: targetCaseId }),
      });

      if (response.ok) {
        evidenceStore.update((items) =>
          items.filter((item) => item.id !== evidenceId)
        );
      }
    } catch (error) {
      console.error("Failed to move evidence:", error);
    }
    closeContextMenu();
  }

  function handleDndConsider(e: CustomEvent, columnId: string) {
    const columnIndex = columns.findIndex((col) => col.id === columnId);
    if (columnIndex !== -1) {
      columns[columnIndex].items = e.detail.items;
    }
  }

  async function handleDndFinalize(e: CustomEvent, columnId: string) {
    const columnIndex = columns.findIndex((col) => col.id === columnId);
    if (columnIndex !== -1) {
      columns[columnIndex].items = e.detail.items;
    }

    // Update evidence status based on column
    const movedItem = e.detail.items.find(
      (item: EvidenceType) => item.id === e.detail.info.id
    );
    if (movedItem && movedItem.status !== columnId) {
      await updateEvidenceStatus(movedItem.id, columnId);
    }
  }

  async function updateEvidenceStatus(evidenceId: string, newStatus: string) {
    try {
      const response = await fetch("/api/evidence/status", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ evidenceId, status: newStatus }),
      });

      if (response.ok) {
        evidenceStore.update((items) => {
          const item = items.find((item) => item.id === evidenceId);
          if (item) {
            item.status = newStatus;
          }
          return items;
        });
      }
    } catch (error) {
      console.error("Failed to update evidence status:", error);
    }
  }

  async function handleFileUpload(files: FileList, columnId: string = "new") {
    for (const file of files) {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("caseId", caseId);
      formData.append("status", columnId);

      try {
        const response = await fetch("/api/evidence/upload", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const newEvidence = await response.json();
          evidenceStore.update((items) => [...items, newEvidence]);
        }
      } catch (error) {
        console.error("Upload failed:", error);
      }
    }
  }

  function handleRealtimeUpdate(data: { type: string; payload: any }) {
    switch (data.type) {
      case "EVIDENCE_POSITION_UPDATE":
        evidenceStore.update((items: EvidenceWithPosition[]) => {
          const item = items.find((item) => item.id === data.payload.id);
          if (item) {
            item.position = { x: data.payload.x, y: data.payload.y };
          }
          return items;
        });
        break;

      case "EVIDENCE_UPDATED":
        evidenceStore.update((items) => {
          const index = items.findIndex((item) => item.id === data.payload.id);
          if (index !== -1) {
            items[index] = { ...items[index], ...data.payload };
          } else {
            items.push(data.payload);
          }
          return items;
        });
        break;

      case "EVIDENCE_DELETED":
        evidenceStore.update((items) =>
          items.filter((item) => item.id !== data.payload.id)
        );
        break;

      case "USER_JOINED":
        activeUsers.update((users) => [...users, data.payload]);
        break;

      case "USER_LEFT":
        activeUsers.update((users) =>
          users.filter((user) => user.id !== data.payload.id)
        );
        break;
    }
  }

  function broadcastPositionUpdate(evidenceId: string, x: number, y: number) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "EVIDENCE_POSITION_UPDATE",
          payload: { id: evidenceId, x, y },
        })
      );
    }
  }

  function switchViewMode(mode: "columns" | "canvas") {
    viewMode = mode;
  }
</script>

<svelte:window on:click={() => closeContextMenu()} />

<div class="w-full h-full min-h-screen bg-background">
  <!-- Header -->
  <Card class="mb-6">
    <CardHeader>
      <div class="flex justify-between items-center">
        <div class="flex items-center gap-4">
          <div class="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center">
            <span class="text-2xl">🕵️</span>
          </div>
          <div>
            <CardTitle class="text-2xl">Detective Board</CardTitle>
            <p class="text-muted-foreground">Case Evidence Management System</p>
          </div>
        </div>
        
        <div class="flex items-center gap-4">
          <!-- View Mode Switcher -->
          <div class="flex gap-2">
            <Button
              variant={viewMode === "columns" ? "default" : "outline"}
              size="sm"
              onclick={() => switchViewMode("columns")}
            >
              <span class="mr-2">📋</span>
              Columns
            </Button>
            <Button
              variant={viewMode === "canvas" ? "default" : "outline"}
              size="sm"
              onclick={() => switchViewMode("canvas")}
            >
              <span class="mr-2">🎨</span>
              Canvas
            </Button>
          </div>
          
          <!-- Active Users -->
          {#if $activeUsers.length > 0}
            <div class="flex items-center gap-2">
              <div class="flex -space-x-2">
                {#each $activeUsers.slice(0, 3) as user}
                  <div class="w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-sm font-medium border-2 border-background">
                    {user.name?.charAt(0) || user.email?.charAt(0) || "?"}
                  </div>
                {/each}
                {#if $activeUsers.length > 3}
                  <div class="w-8 h-8 bg-muted text-muted-foreground rounded-full flex items-center justify-center text-sm border-2 border-background">
                    +{$activeUsers.length - 3}
                  </div>
                {/if}
              </div>
              <Badge variant="outline">{$activeUsers.length} online</Badge>
            </div>
          {/if}
          
          <Button size="sm">
            <span class="mr-2">➕</span>
            New Case
          </Button>
        </div>
      </div>
    </CardHeader>
  </Card>

  <!-- Main Board Area -->
  <main class="flex-1">
    {#if viewMode === "columns"}
      <!-- Columns Container -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 h-full">
        {#each columns as column (column.id)}
          <Card class="h-fit">
            <CardHeader class="pb-3">
              <div class="flex justify-between items-center">
                <CardTitle class="text-lg flex items-center gap-2">
                  <div class="w-3 h-3 bg-primary rounded-full"></div>
                  {column.title}
                </CardTitle>
                <Badge variant="secondary">
                  {column.items.length}
                </Badge>
              </div>
            </CardHeader>
            
            <CardContent class="space-y-4">
              <!-- Upload Zone for first column -->
              {#if column.id === "new"}
                <UploadZone
                  on:upload={(e) => handleFileUpload(e.detail, column.id)}
                />
              {/if}
              
              <!-- Evidence Items -->
              <div
                class="space-y-3 min-h-[200px]"
                use:dndzone={{
                  items: column.items,
                  flipDurationMs: 200,
                  dropTargetStyle: {
                    background: "hsl(var(--muted))",
                    border: "2px dashed hsl(var(--primary))",
                    borderRadius: "8px"
                  },
                }}
                on:consider={(e) => handleDndConsider(e, column.id)}
                on:finalize={(e) => handleDndFinalize(e, column.id)}
              >
                {#each column.items as item (item.id)}
                  <div
                    class="cursor-grab active:cursor-grabbing transition-transform hover:scale-105"
                    on:contextmenu={(e) => handleRightClick(e, item)}
                    role="button"
                    tabindex="0"
                  >
                    <EvidenceCard {item} />
                  </div>
                {/each}
              </div>
            </CardContent>
          </Card>
        {/each}
      </div>
    {:else}
      <!-- Canvas Container -->
      <div
        bind:this={canvasContainer}
        class="relative w-full h-[calc(100vh-200px)] bg-muted/20 rounded-lg border overflow-hidden"
        style="background-image: radial-gradient(circle, hsl(var(--muted-foreground)) 1px, transparent 1px); background-size: 20px 20px;"
      >
        <!-- Canvas Toolbar -->
        <div class="absolute top-4 left-4 flex gap-2 z-10">
          <Button size="sm" variant="outline" title="Reset View">
            <span>🔄</span>
          </Button>
          <Button size="sm" variant="outline" title="Zoom In">
            <span>🔍</span>
          </Button>
          <Button size="sm" variant="outline" title="Zoom Out">
            <span>🔍</span>
          </Button>
          <Button size="sm" variant="outline" title="Add Note">
            <span>📝</span>
          </Button>
          <Button size="sm" variant="outline" title="Add Connection">
            <span>🔗</span>
          </Button>
        </div>
        
        <!-- Evidence Nodes on Canvas -->
        <div class="absolute inset-0">
          {#each canvasEvidence as evidence (evidence.id)}
            <div class="absolute" style="left: {evidence.position.x}px; top: {evidence.position.y}px;">
              <EvidenceNode
                title={evidence.title}
                fileUrl={evidence.fileUrl}
                position={evidence.position}
                size={{ width: 300, height: 200 }}
                isSelected={false}
                isDirty={false}
                on:positionUpdate={(e) =>
                  broadcastPositionUpdate(evidence.id, e.detail.x, e.detail.y)}
              />
            </div>
          {/each}
        </div>
        
        <!-- Canvas Upload Zone -->
        <div class="absolute bottom-4 right-4">
          <UploadZone
            minimal={true}
            on:upload={(e) => handleFileUpload(e.detail, "new")}
          />
        </div>
      </div>
    {/if}
  </main>
</div>

<!-- Context Menu -->
{#if contextMenu.show}
  <ContextMenu
    x={contextMenu.x}
    y={contextMenu.y}
    item={contextMenu.item}
    on:sendToCase={(e) => sendToCase(contextMenu.item?.id || '', e.detail.caseId)}
    on:close={closeContextMenu}
  />
{/if}

<style>
  :global(.dnd-item) {
    cursor: grab;
  }
  :global(.dnd-item:active) {
    cursor: grabbing;
  }
</style>
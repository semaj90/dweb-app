<script lang="ts">
  import type { Evidence } from "$lib/types/api";
  import { onDestroy, onMount } from "svelte";
  import { dndzone } from "svelte-dnd-action";
  import { writable } from "svelte/store";
  import EvidenceNode from "../canvas/EvidenceNode.svelte";
  import ContextMenu from "./ContextMenu.svelte";
  import EvidenceCard from "./EvidenceCard.svelte";
  import UploadZone from "./UploadZone.svelte";

  export let caseId: string;
  export let evidence: EvidenceType[] = [];

  // View modes: 'columns' | 'canvas'
  let viewMode = "columns";

  // Patch: Extend EvidenceType to include position for UI state

  type EvidenceWithPosition = EvidenceType & {
    position: { x: number; y: number };
  };

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
      position: (item as any).position || { x: 100, y: 100 }, // Patch for UI
    })) as EvidenceWithPosition[]
  );

  // Active users in real-time collaboration
  const activeUsers = writable<unknown[]>([]);

  // Canvas view state
  let canvasContainer: HTMLElement;
  let canvasEvidence: EvidenceWithPosition[] = [];

  // Column layout
  let columns: Array<{ id: string; title: string; items: Evidence[] }> = [
    { id: "new", title: "New Evidence", items: [] },
    { id: "reviewing", title: "Under Review", items: [] },
    { id: "approved", title: "Case Ready", items: [] },
  ];

  // Context menu state
  let contextMenu: {
    show: boolean;
    x: number;
    y: number;
    item: Evidence | null;
  } = {
    show: false,
    x: 0,
    y: 0,
    item: null,
  };

  // Drag and drop state
  let draggedItem: Evidence | null = null;
  let ws: WebSocket | null = null;

  // Subscribe to evidence changes
  $: {
    if (viewMode === "canvas") {
      canvasEvidence = $evidenceStore;
    } else {
      distributeEvidence();
    }
  }

  // 1. Extract WebSocket connection logic
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
    // Distribute evidence into columns for initial view
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

  // 2. Fix Evidence type in handleRightClick
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
        // Remove from current view
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
    columns[columnIndex].items = e.detail.items;
  }

  async function handleDndFinalize(e: CustomEvent, columnId: string) {
    const columnIndex = columns.findIndex((col) => col.id === columnId);
    columns[columnIndex].items = e.detail.items;

    // Update evidence status based on column
    const movedItem = e.detail.items.find(
      (item: Evidence) => item.id === e.detail.info.id
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

  // 3. Fix updatePosition usage in handleRealtimeUpdate
  function handleRealtimeUpdate(data: { type: string; payload: unknown }) {
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

  function handleStatusChange(item: EvidenceType, newStatus: string) {
    evidenceStore.update((evidenceList) =>
      evidenceList.map((ev) =>
        ev.id === item.id ? { ...ev, status: newStatus } : ev
      )
    );
  }
</script>

<svelte:window onclick={() => closeContextMenu()} />

<div class="mx-auto px-4 max-w-7xl">
  <!-- Header -->
  <header
    class="mx-auto px-4 max-w-7xl"
  >
    <div class="mx-auto px-4 max-w-7xl">
      <div
        class="mx-auto px-4 max-w-7xl"
      >
        <i class="mx-auto px-4 max-w-7xl"></i>
      </div>
      <div>
        <h1 class="mx-auto px-4 max-w-7xl">Detective Mode</h1>
        <p class="mx-auto px-4 max-w-7xl">Case Evidence Management</p>
      </div>
    </div>
    <div class="mx-auto px-4 max-w-7xl">
      <!-- View Mode Switcher -->
      <div class="mx-auto px-4 max-w-7xl">
        <button
          onclick={() => switchViewMode("columns")}
          class="mx-auto px-4 max-w-7xl"
        >
          <i class="mx-auto px-4 max-w-7xl"></i>
          Columns
        </button>
        <button
          onclick={() => switchViewMode("canvas")}
          class="mx-auto px-4 max-w-7xl"
        >
          <i class="mx-auto px-4 max-w-7xl"></i>
          Canvas
        </button>
      </div>
      <!-- Active Users -->
      {#if $activeUsers.length > 0}
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            {#each $activeUsers.slice(0, 3) as user}
              <div
                class="mx-auto px-4 max-w-7xl"
              >
                {user.name?.charAt(0) || user.email?.charAt(0) || "?"}
              </div>
            {/each}
            {#if $activeUsers.length > 3}
              <div
                class="mx-auto px-4 max-w-7xl"
              >
                +{$activeUsers.length - 3}
              </div>
            {/if}
          </div>
          <span class="mx-auto px-4 max-w-7xl">{$activeUsers.length} online</span
          >
        </div>
      {/if}
      <button
        class="mx-auto px-4 max-w-7xl"
      >
        <i class="mx-auto px-4 max-w-7xl"></i>
        New Case
      </button>
    </div>
  </header>
  <!-- Main Board Area -->
  <main
    class="mx-auto px-4 max-w-7xl"
  >
    {#if viewMode === "columns"}
      <!-- Columns Container -->
      <div
        class="mx-auto px-4 max-w-7xl"
      >
        {#each columns as column (column.id)}
          <section class="mx-auto px-4 max-w-7xl">
            <!-- Column Header -->
            <div class="mx-auto px-4 max-w-7xl">
              <h2 class="mx-auto px-4 max-w-7xl">
                <div
                  class="mx-auto px-4 max-w-7xl"
                ></div>
                {column.title}
              </h2>
              <span
                class="mx-auto px-4 max-w-7xl"
              >
                {column.items.length}
              </span>
            </div>
            <!-- Upload Zone for first column -->
            {#if column.id === "new"}
              <UploadZone
                on:upload={(e) => handleFileUpload(e.detail, column.id)}
              />
            {/if}
            <!-- Evidence Items -->
            <div
              class="mx-auto px-4 max-w-7xl"
              use:dndzone={{
                items: column.items,
                flipDurationMs: 200,
                dropTargetStyle: {
                  background: "var(--drop-zone-active)",
                  border: "2px dashed var(--drop-indicator)",
                },
              "
              on:consider={(e) => handleDndConsider(e, column.id)}
              on:finalize={(e) => handleDndFinalize(e, column.id)}
            >
              {#each column.items as item (item.id)}
                <div
                  class="mx-auto px-4 max-w-7xl"
                  on:contextmenu={(e) => handleRightClick(e, item)}
                  role="menuitem"
                  tabindex={0}
                >
                  <EvidenceCard {item} />
                </div>
              {/each}
            </div>
          </section>
        {/each}
      </div>
    {:else}
      <!-- Canvas Container -->
      <div
        bind:this={canvasContainer}
        class="mx-auto px-4 max-w-7xl"
        style="background-image: radial-gradient(circle, #e5e7eb 1px, transparent 1px); background-size: 20px 20px;"
      >
        <!-- Canvas Toolbar -->
        <div
          class="mx-auto px-4 max-w-7xl"
        >
          <button
            class="mx-auto px-4 max-w-7xl"
            aria-label="Reset View"
            title="Reset View"
          >
            <i class="mx-auto px-4 max-w-7xl"></i>
          </button>
          <button
            class="mx-auto px-4 max-w-7xl"
            aria-label="Zoom In"
            title="Zoom In"
          >
            <i class="mx-auto px-4 max-w-7xl"></i>
          </button>
          <button
            class="mx-auto px-4 max-w-7xl"
            aria-label="Zoom Out"
            title="Zoom Out"
          >
            <i class="mx-auto px-4 max-w-7xl"></i>
          </button>
          <button
            class="mx-auto px-4 max-w-7xl"
            aria-label="Add Text Note"
            title="Add Text Note"
          >
            <i class="mx-auto px-4 max-w-7xl"></i>
          </button>
          <button
            class="mx-auto px-4 max-w-7xl"
            aria-label="Add Connection"
            title="Add Connection"
          >
            <i class="mx-auto px-4 max-w-7xl"></i>
          </button>
        </div>
        <!-- Evidence Nodes on Canvas -->
        <div class="mx-auto px-4 max-w-7xl">
          {#each canvasEvidence as evidence (evidence.id)}
            <div
              class="mx-auto px-4 max-w-7xl"
            >
              <EvidenceNode
                title={evidence.title}
                fileUrl={evidence.fileUrl}
                position={evidence.position}
                size={{ width: 400, height: 300  "
                isSelected={false}
                isDirty={false}
                on:positionUpdate={(e) =>
                  broadcastPositionUpdate(evidence.id, e.detail.x, e.detail.y)}
              />
            </div>
          {/each}
        </div>
        <!-- Canvas Upload Zone -->
        <div class="mx-auto px-4 max-w-7xl">
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
    on:sendToCase={(e) => sendToCase(contextMenu.item?.id, e.detail.caseId)}
    on:close={closeContextMenu}
  />
{/if}

<style>
  .detective-board {
    font-family: var(--font-family);
  }

  .evidence-item {
    transform: translateZ(0); /* Force hardware acceleration */
  }

  :global(.dnd-item) {
    cursor: grab;
  }

  :global(.dnd-item:active) {
    cursor: grabbing;
  }
</style>

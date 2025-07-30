<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { dndzone } from "svelte-dnd-action";
  import { writable } from "svelte/store";
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/Card";
  import Button from "$lib/components/ui/button/Button.svelte";
  import Badge from "$lib/components/ui/Badge.svelte";
  import EvidenceNode from "../canvas/EvidenceNode.svelte";
import * as ContextMenu from "$lib/components/ui/context-menu";
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


  // Import context menu store and actions
  import { contextMenuStore, contextMenuActions } from "$lib/stores";
  import { get } from "svelte/store";

  // Subscribe to context menu state
  let contextMenu = get(contextMenuStore);
  const unsubscribeContextMenu = contextMenuStore.subscribe((v) => (contextMenu = v));

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
    contextMenuActions.open(event.clientX, event.clientY, item);
  }


  function closeContextMenu() {
    contextMenuActions.close();
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

  // Additional imports moved from outside script tag
  import { keyboardShortcuts } from "$lib/stores";
  import { get as getStore } from "svelte/store";

  // Listen for global keyboard shortcuts (AI-driven)
  function handleGlobalKeydown(event: KeyboardEvent) {
    // Ignore if typing in input/textarea
    const target = event.target as HTMLElement;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;

    // Format key combo (e.g., Ctrl+I)
    const parts = [];
    if (event.ctrlKey || event.metaKey) parts.push("Ctrl");
    if (event.altKey) parts.push("Alt");
    if (event.shiftKey) parts.push("Shift");
    const key = event.key.length === 1 ? event.key.toUpperCase() : event.key;

    const combo = parts.length > 0 ? `${parts.join("+")}+${key}` : key;

    // Check shortcuts and execute if found
    const shortcuts = getStore(keyboardShortcuts);
    const shortcut = shortcuts.find(s => s.key === combo);
    if (shortcut && shortcut.action) {
      event.preventDefault();
      shortcut.action();
    }
  }

  // Wire up shortcut actions on mount
  onMount(() => {
    // Example: Attach context menu open to Ctrl+I shortcut
    keyboardShortcuts.update(shortcuts =>
      shortcuts.map(s =>
        s.key === "Ctrl+I"
          ? {
              ...s,
              action: () => {
                // Open context menu for first evidence in current view (or selected)
                let item = null;
                if (viewMode === "columns") {
                  for (const col of columns) {
                    if (col.items.length > 0) { item = col.items[0]; break; }
                  }
                } else if (canvasEvidence.length > 0) {
                  item = canvasEvidence[0];
                }
                if (item) {
                  const x = window.innerWidth / 2;
                  const y = window.innerHeight / 2;
                  contextMenuActions.open(x, y, item);
                }
              }
            }
          : s
      )
    );
  });

  onDestroy(() => {
    if (ws) {
      ws.close();
    }
    unsubscribeContextMenu();
  });

  // Mini-modal state
  let miniModal: { show: boolean; x: number; y: number; type: string } = { show: false, x: 0, y: 0, type: '' };

  function showMiniModal(type: string) {
    miniModal = { show: true, x: contextMenu.x + 180, y: contextMenu.y, type };
  }
  function hideMiniModal() {
    miniModal = { show: false, x: 0, y: 0, type: '' };
  }

  // Save to logic: update user activity store, backend, Qdrant, Loki.js, MCP, LLM
  import { getContextAwareSuggestions, callContext7Tool } from "$lib/ai/mcp-helpers";
  import { page } from '$app/stores';
  import Fuse from 'fuse.js';

  async function saveTo(target: string) {
    if (!contextMenu.item) return closeContextMenu();
    // 1. Update user activity store (local)
    // (Assume a store or API exists, e.g., /api/user-activity)
    try {
      await fetch('/api/user-activity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: $page.data.user?.id,
          evidenceId: contextMenu.item.id,
          action: 'save',
          target
        })
      });
    } catch (e) { console.warn('User activity store update failed', e); }

    // 2. Qdrant/Loki.js: (stub, replace with real integration)
    try {
      await fetch('/api/vector-store', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evidence: contextMenu.item,
          target
        })
      });
    } catch (e) { console.warn('Qdrant/Loki.js update failed', e); }

    // 3. MCP/Context7: add to memory/context
    try {
      await callContext7Tool('mcp_memory2_add_observations', {
        observations: [{
          contents: [JSON.stringify(contextMenu.item)],
          entityName: target
        }]
      });
    } catch (e) { console.warn('MCP memory add failed', e); }

    // 4. Optionally trigger LLM retrain or context update
    try {
      await callContext7Tool('call-llm-context-update', {
        evidenceId: contextMenu.item.id,
        target
      });
    } catch (e) { /* ignore if not implemented */ }

    closeContextMenu();
  }

  // Find/search logic: Fuse.js, Qdrant, MCP, LLM autocomplete
  async function findWithLLM() {
    openFindModal();
  }

  let findModal = { show: false, query: '', results: [], loading: false, error: '', suggestions: [] };

  function openFindModal() {
    findModal = { show: true, query: contextMenu.item?.title || '', results: [], loading: false, error: '', suggestions: [] };
  }
  function closeFindModal() {
    findModal = { ...findModal, show: false };
    closeContextMenu();
  }

  async function runFindSearch() {
    if (!contextMenu.item) return closeFindModal();
    findModal.loading = true;
    findModal.error = '';
    findModal.results = [];
    findModal.suggestions = [];
    // 1. Local fuzzy search (Fuse.js)
    try {
      const items = $evidenceStore;
      const fuse = new Fuse(items, { keys: ['title', 'description', 'tags'] });
      findModal.results = fuse.search(findModal.query || contextMenu.item.title || '');
    } catch (e) { findModal.error = 'Local search failed'; }
    // 2. Qdrant/Vector search
    try {
      const resp = await fetch('/api/vector-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: findModal.query || contextMenu.item.title })
      });
      const vectorResults = await resp.json();
      findModal.results = [...findModal.results, ...vectorResults];
    } catch (e) { findModal.error += ' Qdrant search failed.'; }
    // 3. MCP/LLM autocomplete
    try {
      const suggestions = await getContextAwareSuggestions({
        workspaceRoot: '',
        activeFiles: [],
        errors: [],
        userIntent: 'feature-development',
        recentPrompts: [findModal.query || contextMenu.item.title],
        projectType: 'sveltekit-legal-ai'
      });
      findModal.suggestions = suggestions;
    } catch (e) { findModal.error += ' LLM autocomplete failed.'; }
    findModal.loading = false;
  }
</script>

<svelte:window onclick={() => closeContextMenu()} onkeydown={handleGlobalKeydown} />

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
  <ContextMenu.Root>
    <ContextMenu.Content style={`position:fixed;left:${contextMenu.x}px;top:${contextMenu.y}px;`} menu={true}>
      <div class="container mx-auto px-4">
        <p class="container mx-auto px-4">Evidence Actions</p>
      </div>
      <ContextMenu.Item on:select={() => window.open(`/evidence/${contextMenu.item?.id}`, "_blank")}>View Details</ContextMenu.Item>
      <ContextMenu.Item on:select={() => window.location.href = `/evidence/${contextMenu.item?.id}/edit`}>Edit</ContextMenu.Item>
      <ContextMenu.Item on:select={() => contextMenu.item?.fileUrl && (() => { const link = document.createElement('a'); link.href = contextMenu.item.fileUrl; link.download = contextMenu.item.fileName || 'evidence'; link.click(); })()}>Download</ContextMenu.Item>
      <ContextMenu.Item on:select={() => console.log('Duplicate evidence:', contextMenu.item?.id)}>Duplicate</ContextMenu.Item>
      <ContextMenu.Item on:select={() => {/* TODO: Audit evidence logic */}}>Audit (Semantic/Vector)</ContextMenu.Item>
      <ContextMenu.Item on:select={() => {/* TODO: Trigger agent review logic */}}>Trigger Agent Review</ContextMenu.Item>
      <ContextMenu.Separator />
      <div class="container mx-auto px-4"><p class="container mx-auto px-4">Save/Link</p></div>
      <ContextMenu.Item on:mouseenter={() => showMiniModal('citation')} on:mouseleave={hideMiniModal} on:select={() => saveTo('savedcitations')}>Add to /savedcitations</ContextMenu.Item>
      <ContextMenu.Item on:mouseenter={() => showMiniModal('aisummary')} on:mouseleave={hideMiniModal} on:select={() => saveTo('savedaisummaries')}>Add to /savedaisummaries</ContextMenu.Item>
      <ContextMenu.Item on:mouseenter={() => showMiniModal('userreport')} on:mouseleave={hideMiniModal} on:select={() => saveTo('saveduserreports')}>Add to /saveduserreports</ContextMenu.Item>
      <ContextMenu.Item on:mouseenter={() => showMiniModal('mcpcontext')} on:mouseleave={hideMiniModal} on:select={() => saveTo('mcpcontext')}>Add to MCP Context (LLM)</ContextMenu.Item>
      <ContextMenu.Separator />
      <div class="container mx-auto px-4"><p class="container mx-auto px-4">Send to Case</p></div>
      <ContextMenu.Item on:select={() => {/* TODO: Show case selection and call sendToCase */}}>Send to /casesid</ContextMenu.Item>
      <ContextMenu.Separator />
      <ContextMenu.Item on:mouseenter={() => showMiniModal('find')} on:mouseleave={hideMiniModal} on:select={findWithLLM}>Find (search/query with LLM)</ContextMenu.Item>
      {#if findModal.show}
        <div class="fixed z-50 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-background border border-primary rounded-lg shadow-lg p-6 w-full max-w-lg">
          <div class="flex flex-col gap-4">
            <div class="flex items-center gap-2">
              <input
                class="flex-1 border rounded px-3 py-2 text-base bg-muted text-foreground focus:outline-none focus:ring focus:border-primary"
                type="text"
                bind:value={findModal.query}
                placeholder="Enter keywords or question..."
                on:keydown={(e) => { if (e.key === 'Enter') runFindSearch(); }}
                autofocus
              />
              <Button size="sm" on:click={runFindSearch} disabled={findModal.loading}>Search</Button>
              <Button size="sm" variant="outline" on:click={closeFindModal}>Close</Button>
            </div>
            {#if findModal.loading}
              <div class="text-primary">Searching...</div>
            {/if}
            {#if findModal.error}
              <div class="text-red-500">{findModal.error}</div>
            {/if}
            {#if findModal.results.length > 0}
              <div>
                <div class="font-semibold mb-2">Results:</div>
                <ul class="space-y-2 max-h-48 overflow-y-auto">
                  {#each findModal.results as result, i}
                    <li class="p-2 rounded hover:bg-muted cursor-pointer border-b border-muted-foreground/10">
                      {result.item ? result.item.title : (result.title || result.text || JSON.stringify(result))}
                    </li>
                  {/each}
                </ul>
              </div>
            {/if}
            {#if findModal.suggestions.length > 0}
              <div>
                <div class="font-semibold mb-2">AI Suggestions:</div>
                <ul class="space-y-2">
                  {#each findModal.suggestions as suggestion}
                    <li class="p-2 rounded bg-muted/50 border border-primary/10">
                      <div class="font-medium">{suggestion.expectedOutput}</div>
                      <div class="text-xs text-muted-foreground">{suggestion.reasoning}</div>
                    </li>
                  {/each}
                </ul>
              </div>
            {/if}
          </div>
        </div>
      {/if}
    </ContextMenu.Content>
  </ContextMenu.Root>
{/if}

<!-- Mini-modal for context menu hover -->
{#if miniModal.show}
  <div class="fixed z-50 p-2 bg-gray-900 text-white rounded shadow-lg" style="left: {miniModal.x}px; top: {miniModal.y}px; min-width: 180px;">
    {#if miniModal.type === 'citation'}
      <div>Save this evidence as a citation. Will be available for search and LLM context.</div>
    {:else if miniModal.type === 'aisummary'}
      <div>Save an AI-generated summary for this evidence. Used for quick reference and LLM context.</div>
    {:else if miniModal.type === 'userreport'}
      <div>Save a user report or annotation for this evidence. Used for personalization and LLM context.</div>
    {:else if miniModal.type === 'mcpcontext'}
      <div>Add this evidence to the MCP context for the local LLM. Used for personalized completions and search.</div>
    {:else if miniModal.type === 'find'}
      <div>Find related items using keywords, Fuse.js fuzzy search, and LLM autocomplete from your activity store.</div>
    {/if}
  </div>
{/if}
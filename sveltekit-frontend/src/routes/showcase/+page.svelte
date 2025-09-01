<!--
=================================================================
  FIXED: NieR & Harvard-Themed Svelte Components Showcase
  Integrated with enhanced UI components and vector search
=================================================================
-->

<script lang="ts">
  // Updated to use melt-ui components
  import Dialog from '$lib/components/ui/MeltDialog.svelte';
  
  // TODO: Replace with melt-ui equivalent when available
  // import { DropdownMenu } from 'bits-ui';
  import { onMount, tick } from 'svelte';
  import type { Case, Evidence, Report, CanvasState } from '$lib/data/types';
  import type { ChatMessage } from '$lib/types/chat';
  import { notifications } from '$lib/stores/notification';
  import { aiService } from '$lib/services/aiService';
  // Import client-safe vector search - use API endpoints instead
  // import { vectorService } from '$lib/server/vector/EnhancedVectorService';
  import {
    ArrowDown,
    CornerDownLeft,
    Trash2,
    Bot,
    User,
    LayoutDashboard,
    FileText,
    Users,
    Scale,
    UploadCloud,
    LifeBuoy,
    Settings,
    FileQuestion,
    Sun,
    Moon,
    Search,
    Bell,
    MoreHorizontal,
    PlusCircle,
    BrainCircuit,
    ShieldCheck,
    BarChart3,
  } from 'lucide-svelte';

  // Import our enhanced UI components
  import Button from '$lib/components/ui/Button.svelte';
  import { Card } from '$lib/components/ui/card';
  import Input from '$lib/components/ui/Input.svelte';
  import Modal from '$lib/components/ui/Modal.svelte';

  // State for demonstrations
  let modalOpen = $state(false);
  let searchQuery = $state('');
  let vectorResults = $state([]);
  let isSearching = $state(false);

  // Demo data
let layoutData = $state({
    user: { name: 'James', email: 'james@example.com' },
    stats: { totalCases: 12, openCases: 5, closedCases: 7, evidenceCount: 142 },
    recentActivity: [
      { action: 'Uploaded Evidence', details: 'witness_statement_01.pdf', time: '2m ago' },
      { action: 'Updated Case', details: 'State v. Anderson', time: '1h ago' },
      { action: 'Generated Report', details: 'Initial Analysis', time: '3h ago' },
    ],
  });

  // Vector search integration
  async function performVectorSearch() {
    if (!searchQuery.trim()) return;

    isSearching = true;
    try {
      const response = await fetch('/api/vector/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          options: { limit: 5, threshold: 0.7 },
        }),
      });

      const data = await response.json();
      vectorResults = data.results || [];

      notifications.add({
        type: 'success',
        title: 'Search Complete',
        message: `Found ${vectorResults.length} results`,
      });
    } catch (error) {
      console.error('Vector search failed:', error);
      notifications.add({
        type: 'error',
        title: 'Search Failed',
        message: 'Vector search service unavailable',
      });
    } finally {
      isSearching = false;
    }
  }
</script>

<div class="p-8 font-sans bg-nier-surface text-nier-white min-h-screen">
  <h1 class="text-3xl font-bold mb-4 border-b-2 border-crimson nier-text-glow">
    Enhanced Legal AI Showcase
  </h1>

  <!-- Vector Search Demo -->
  <section class="mb-12">
    <h2 class="text-2xl font-semibold mb-4 text-gold">Vector Search Integration</h2>
    <div class="nier-card nier-card-interactive p-6">
      <div class="flex gap-4 mb-4">
        <Input
          bind:value={searchQuery}
          placeholder="Search cases, evidence, legal documents..."
          class="flex-1" />
        <Button on:on:click={performVectorSearch} loading={isSearching} disabled={!searchQuery.trim()}>
          <Search class="w-5 h-5 mr-2" />
          Search
        </Button>
      </div>

      {#if vectorResults.length > 0}
        <div class="mt-4">
          <h4 class="text-lg font-semibold text-gold mb-2">Search Results:</h4>
          <div class="space-y-2">
            {#each vectorResults as result}
              <div class="p-3 bg-nier-surface-light rounded border border-nier-border">
                <div class="flex justify-between items-start">
                  <div>
                    <h5 class="font-semibold text-nier-white">
                      {result.metadata?.title || 'Untitled'}
                    </h5>
                    <p class="text-sm text-nier-text-muted">{result.content?.slice(0, 100)}...</p>
                    <span class="text-xs text-nier-accent"
                      >Score: {(result.score * 100).toFixed(1)}%</span>
                  </div>
                  <span class="badge status-{result.metadata?.type || 'default'}"
                    >{result.metadata?.type || 'document'}</span>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  </section>

  <!-- Enhanced UI Components Demo -->
  <section class="mb-12">
    <h2 class="text-2xl font-semibold mb-4 text-gold">Enhanced UI Components</h2>

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <!-- Button variants -->
      <Card>
        <div class="p-4">
          <h3 class="text-lg font-semibold mb-4 text-crimson">Button Variants</h3>
          <div class="space-y-3">
            <Button variant="primary">Primary Action</Button>
            <Button variant="secondary">Secondary Action</Button>
            <Button variant="ghost">Ghost Button</Button>
            <Button variant="danger">Delete Action</Button>
          </div>
        </div>
      </Card>

      <!-- Modal demo -->
      <Card>
        <div class="p-4">
          <h3 class="text-lg font-semibold mb-4 text-crimson">Modal Component</h3>
          <Button on:on:click={() => (modalOpen = true)}>Open Modal</Button>

          <Modal bind:open={modalOpen} title="System Alert">
            <div class="mt-4">
              <p class="text-nier-light-gray mb-4">
                This modal uses the enhanced Modal component with Svelte 5 runes. The modal
                integrates with bits-ui and follows Svelte 5 best practices.
              </p>
              <div class="flex gap-2 justify-end">
                <Button variant="ghost" on:on:click={() => (modalOpen = false)}>Cancel</Button>
                <Button on:on:click={() => (modalOpen = false)}>Acknowledge</Button>
              </div>
            </div>
          </Modal>
        </div>
      </Card>

      <!-- Input components -->
      <Card>
        <div class="p-4">
          <h3 class="text-lg font-semibold mb-4 text-crimson">Input Components</h3>
          <div class="space-y-3">
            <Input label="Case Title" placeholder="Enter case title..." />
            <Input label="Evidence ID" type="search" placeholder="Search evidence..." />
            <Input
              label="Error Example"
              error="This field is required"
              placeholder="Input with error..." />
          </div>
        </div>
      </Card>
    </div>
  </section>

  <!-- Application Layout Demo -->
  <section class="mb-12">
    <h2 class="text-2xl font-semibold mb-4 text-gold">Application Layout</h2>
    <div class="h-[600px] border border-nier-border rounded-lg overflow-hidden">
      {@render LayoutDemo()}
    </div>
  </section>

  <!-- Integration Status -->
  <section class="mb-12">
    <h2 class="text-2xl font-semibold mb-4 text-gold">Integration Status</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {@render StatusCard({
        title: 'Svelte 5 Runes',
        status: 'active',
        description: 'Using $state and $props',
      })}
      {@render StatusCard({
        title: 'Bits UI',
        status: 'active',
        description: 'Headless components integrated',
      })}
      {@render StatusCard({
        title: 'Vector Search',
        status: 'active',
        description: 'Qdrant + PostgreSQL ready',
      })}
      {@render StatusCard({
        title: 'UnoCSS',
        status: 'active',
        description: 'Utility classes configured',
      })}
    </div>
  </section>
</div>

{#snippet StatusCard({ title, status, description })}
  <Card variant="interactive">
    <div class="p-4 text-center">
      <div
        class="w-12 h-12 mx-auto mb-3 rounded-full flex items-center justify-center {status ===
        'active'
          ? 'bg-green-500/20'
          : 'bg-red-500/20'}">
        {#if status === 'active'}
          <ShieldCheck class="w-6 h-6 text-green-400" />
        {:else}
          <FileQuestion class="w-6 h-6 text-red-400" />
        {/if}
      </div>
      <h3 class="font-semibold text-nier-white">{title}</h3>
      <p class="text-sm text-nier-text-muted mt-1">{description}</p>
      <span
        class="inline-block mt-2 px-2 py-1 text-xs rounded {status === 'active'
          ? 'bg-green-500/20 text-green-400'
          : 'bg-red-500/20 text-red-400'}">
        {status.toUpperCase()}
      </span>
    </div>
  </Card>
{/snippet}

{#snippet LayoutDemo()}
  <div class="flex h-full bg-nier-bg">
    <!-- Sidebar -->
    <aside class="w-64 bg-nier-surface border-r border-nier-border p-4">
      <div class="mb-6">
        <h2 class="text-xl font-bold text-nier-accent">⚖️ DEEDS</h2>
      </div>
      <nav class="space-y-2">
        {#each [{ icon: LayoutDashboard, label: 'Dashboard' }, { icon: FileText, label: 'Cases' }, { icon: Scale, label: 'Evidence' }, { icon: Users, label: 'Users' }] as item}
          {@const IconComponent = item.icon}
          <a
            href="/showcase"
            class="flex items-center gap-3 p-2 rounded hover:bg-nier-surface-light text-nier-text">
            <IconComponent class="w-5 h-5" />
            {item.label}
          </a>
        {/each}
      </nav>
    </aside>

    <!-- Main content -->
    <main class="flex-1 p-6">
      <h1 class="text-2xl font-bold text-nier-white mb-4">
        Welcome back, {layoutData.user.name}
      </h1>

      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {#each [{ title: 'Total Cases', value: layoutData.stats.totalCases, icon: FileText }, { title: 'Open Cases', value: layoutData.stats.openCases, icon: FileText }, { title: 'Closed Cases', value: layoutData.stats.closedCases, icon: FileText }, { title: 'Evidence Items', value: layoutData.stats.evidenceCount, icon: Scale }] as stat}
          {@const StatIcon = stat.icon}
          <Card>
            <div class="p-4">
              <div class="flex justify-between items-center mb-2">
                <h4 class="text-sm font-medium text-nier-text-muted">{stat.title}</h4>
                <StatIcon class="w-5 h-5 text-nier-accent" />
              </div>
              <p class="text-2xl font-bold text-nier-white">{stat.value}</p>
            </div>
          </Card>
        {/each}
      </div>

      <Card>
        <div class="p-6">
          <h3 class="text-lg font-semibold text-nier-white mb-4">Recent Activity</h3>
          <div class="space-y-3">
            {#each layoutData.recentActivity as activity}
              <div class="flex items-center gap-3 p-3 bg-nier-surface-light rounded">
                <div class="w-2 h-2 bg-nier-accent rounded-full"></div>
                <div>
                  <p class="font-medium text-nier-white">{activity.action}</p>
                  <p class="text-sm text-nier-text-muted">{activity.details}</p>
                </div>
                <span class="ml-auto text-xs text-nier-text-muted">{activity.time}</span>
              </div>
            {/each}
          </div>
        </div>
      </Card>
    </main>
  </div>
{/snippet}

<style>
  /* Enhanced Nier theme styles */
  :global(:root) {
    --nier-bg: #0a0a0a;
    --nier-surface: #1a1a1a;
    --nier-surface-light: #2a2a2a;
    --nier-border: #404040;
    --nier-text: #e5e5e5;
    --nier-text-muted: #9ca3af;
    --nier-accent: #f59e0b;
    --nier-accent-light: #fbbf24;
  }

  .nier-text-glow {
    text-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
  }

  .badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    font-weight: 500;
    border-radius: 0.25rem;
  }

  .status-case {
    background-color: rgb(59 130 246 / 0.2);
    color: rgb(96 165 250);
  }

  .status-evidence {
    background-color: rgb(168 85 247 / 0.2);
    color: rgb(196 181 253);
  }

  .status-criminal {
    background-color: rgb(239 68 68 / 0.2);
    color: rgb(248 113 113);
  }

  .status-document {
    background-color: rgb(107 114 128 / 0.2);
    color: rgb(156 163 175);
  }

  .status-default {
    background-color: rgb(107 114 128 / 0.2);
    color: rgb(156 163 175);
  }
</style>

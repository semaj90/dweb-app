

<script lang="ts">
  import { createContextMenu, melt } from '@melt-ui/svelte'
  import { fly, scale } from 'svelte/transition'
import {
    FileText,
    Users,
    Calendar,
    MoreVertical,
    Eye,
    Edit,
    Archive,
    Trash2,
    AlertTriangle,
    Clock,
    CheckCircle
  } from 'lucide-svelte'







  interface CaseData {
    id: string
    title: string
    description?: string
    status: 'active' | 'pending' | 'closed' | 'archived'
    priority: 'critical' | 'high' | 'medium' | 'low'
    created: Date | string
    updated?: Date | string
    assignee?: {
      name: string
      avatar?: string
    }
    stats: {
      evidence: number
      witnesses: number
      documents: number
    }
    tags?: string[]
    progress?: number
  }

  interface Props {
    case: CaseData
    onView?: (id: string) => void
    onEdit?: (id: string) => void
    onArchive?: (id: string) => void
    onDelete?: (id: string) => void
  }

  let {
    case: caseData,
    onView = () => {},
    onEdit = () => {},
    onArchive = () => {},
    onDelete = () => {}
  } = $props()

  // Create context menu
  const {
    elements: { trigger, menu, item, separator },
    states: { open }
  } = createContextMenu({
    forceVisible: true,
  })

  // Status configurations
  const statusConfig = {
    active: {
      icon: CheckCircle,
      class: 'nier-badge-success',
      label: 'Active'
    },
    pending: {
      icon: Clock,
      class: 'nier-badge-warning',
      label: 'Pending'
    },
    closed: {
      icon: CheckCircle,
      class: 'nier-badge-error',
      label: 'Closed'
    },
    archived: {
      icon: Archive,
      class: 'text-nier-gray dark:text-nier-silver',
      label: 'Archived'
    }
  }

  // Priority configurations
  const priorityConfig = {
    critical: {
      class: 'priority-critical',
      icon: '🚨',
      color: 'text-harvard-crimson'
    },
    high: {
      class: 'priority-high',
      icon: '⚠️',
      color: 'text-digital-orange'
    },
    medium: {
      class: 'priority-medium',
      icon: '📌',
      color: 'text-nier-amber'
    },
    low: {
      class: 'priority-low',
      icon: '📍',
      color: 'text-digital-green'
    }
  }

  // Format date
  const formatDate = (date: Date | string) => {
    const d = new Date(date)
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }).format(d)
  }

  // Calculate days ago
  const daysAgo = (date: Date | string) => {
    const d = new Date(date)
    const now = new Date()
    const diff = Math.floor((now.getTime() - d.getTime()) / (1000 * 60 * 60 * 24))
    if (diff === 0) return 'Today'
    if (diff === 1) return 'Yesterday'
    return `${diff} days ago`
  }

  const currentStatus = statusConfig[caseData.status]
  const currentPriority = priorityConfig[caseData.priority]
</script>

<div
  use:melt={$trigger}
  class="case-card {currentPriority.class} group relative overflow-hidden"
  role="article"
  aria-label="Case {caseData.id}"
>
  <!-- Background Pattern -->
  <div class="absolute inset-0 opacity-5 dark:opacity-10 pointer-events-none">
    <div class="absolute inset-0" style="background-image: url('data:image/svg+xml,%3Csvg width=\'40\' height=\'40\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cg fill=\'none\' stroke=\'%23000\' stroke-width=\'0.5\' opacity=\'0.3\'%3E%3Cpath d=\'M0 0h40v40H0z\'/%3E%3Cpath d=\'M0 0l40 40M40 0L0 40\'/%3E%3C/g%3E%3C/svg%3E');"></div>
  </div>

  <!-- Card Content -->
  <div class="relative">
    <!-- Header -->
    <div class="flex items-start justify-between mb-4">
      <div class="flex-1">
        <div class="flex items-center gap-2 mb-1">
          <span class="text-sm font-mono text-nier-gray dark:text-nier-silver">
            {caseData.id}
          </span>
          <span class="{currentPriority.color} text-lg" title="{caseData.priority} priority">
            {currentPriority.icon}
          </span>
        </div>

        <h3 class="text-lg font-semibold nier-heading line-clamp-1 group-hover:text-harvard-crimson dark:group-hover:text-digital-green nier-transition">
          {caseData.title}
        </h3>

        {#if caseData.description}
          <p class="text-sm text-nier-gray dark:text-nier-silver line-clamp-2 mt-1">
            {caseData.description}
          </p>
        {/if}
      </div>

      <div class="flex items-center gap-2">
        <span class="{currentStatus.class}">
          {#key currentStatus.icon}
            <currentStatus.icon class="w-3 h-3 mr-1" />
          {/key}
          {currentStatus.label}
        </span>

        <button
          class="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-nier-white/50 dark:hover:bg-nier-black/50 nier-transition"
          aria-label="More options"
        >
          <MoreVertical class="w-5 h-5 text-nier-gray dark:text-nier-silver" />
        </button>
      </div>
    </div>

    <!-- Stats Grid -->
    <div class="grid grid-cols-3 gap-4 mb-4">
      <div class="text-center p-3 rounded-lg bg-nier-white/50 dark:bg-nier-black/50">
        <div class="flex items-center justify-center gap-1 mb-1">
          <FileText class="w-4 h-4 text-nier-gray dark:text-nier-silver" />
          <p class="text-xl font-bold text-harvard-crimson dark:text-digital-green">
            {caseData.stats.documents}
          </p>
        </div>
        <p class="text-xs text-nier-gray dark:text-nier-silver">Documents</p>
      </div>

      <div class="text-center p-3 rounded-lg bg-nier-white/50 dark:bg-nier-black/50">
        <div class="flex items-center justify-center gap-1 mb-1">
          <AlertTriangle class="w-4 h-4 text-nier-gray dark:text-nier-silver" />
          <p class="text-xl font-bold text-harvard-crimson dark:text-digital-green">
            {caseData.stats.evidence}
          </p>
        </div>
        <p class="text-xs text-nier-gray dark:text-nier-silver">Evidence</p>
      </div>

      <div class="text-center p-3 rounded-lg bg-nier-white/50 dark:bg-nier-black/50">
        <div class="flex items-center justify-center gap-1 mb-1">
          <Users class="w-4 h-4 text-nier-gray dark:text-nier-silver" />
          <p class="text-xl font-bold text-harvard-crimson dark:text-digital-green">
            {caseData.stats.witnesses}
          </p>
        </div>
        <p class="text-xs text-nier-gray dark:text-nier-silver">Witnesses</p>
      </div>
    </div>

    <!-- Progress Bar (if applicable) -->
    {#if caseData.progress !== undefined}
      <div class="mb-4">
        <div class="flex justify-between items-center mb-1">
          <span class="text-xs text-nier-gray dark:text-nier-silver">Progress</span>
          <span class="text-xs font-medium">{caseData.progress}%</span>
        </div>
        <div class="h-2 bg-nier-white/50 dark:bg-nier-black/50 rounded-full overflow-hidden">
          <div
            class="h-full nier-gradient-digital nier-transition"
            style="width: {caseData.progress}%"
          ></div>
        </div>
      </div>
    {/if}

    <!-- Tags -->
    {#if caseData.tags && caseData.tags.length > 0}
      <div class="flex flex-wrap gap-2 mb-4">
        {#each caseData.tags as tag}
          <span class="text-xs px-2 py-1 rounded-full bg-nier-white/50 dark:bg-nier-black/50 text-nier-gray dark:text-nier-silver">
            #{tag}
          </span>
        {/each}
      </div>
    {/if}

    <!-- Footer -->
    <div class="flex items-center justify-between pt-4 border-t border-nier-light-gray dark:border-nier-gray/30">
      <div class="flex items-center gap-3">
        {#if caseData.assignee}
          <div class="flex items-center gap-2">
            {#if caseData.assignee.avatar}
              <img
                src={caseData.assignee.avatar}
                alt={caseData.assignee.name}
                class="w-6 h-6 rounded-full"
              />
            {:else}
              <div class="w-6 h-6 rounded-full bg-nier-gradient-crimson flex items-center justify-center">
                <span class="text-xs font-bold text-nier-white">
                  {caseData.assignee.name.charAt(0).toUpperCase()}
                </span>
              </div>
            {/if}
            <span class="text-sm text-nier-gray dark:text-nier-silver">
              {caseData.assignee.name}
            </span>
          </div>
        {/if}
      </div>

      <div class="flex items-center gap-2 text-xs text-nier-gray dark:text-nier-silver">
        <Calendar class="w-3 h-3" />
        <span title={formatDate(caseData.created)}>
          {daysAgo(caseData.created)}
        </span>
      </div>
    </div>

    <!-- Action Buttons -->
    <div class="flex gap-2 mt-4">
      <button
        onclick={() => onView(caseData.id)}
        class="nier-button-primary text-sm px-4 py-2 flex-1"
      >
        <Eye class="w-4 h-4" />
        View Details
      </button>
      <button
        onclick={() => onEdit(caseData.id)}
        class="nier-button-outline text-sm px-4 py-2"
      >
        <Edit class="w-4 h-4" />
      </button>
    </div>
  </div>

  <!-- Digital Effect on Hover -->
  <div class="absolute inset-0 bg-gradient-to-br from-transparent to-digital-green/5 opacity-0 group-hover:opacity-100 pointer-events-none nier-transition"></div>
</div>

<!-- Context Menu -->
{#if $open}
  <div
    use:melt={$menu}
    class="nier-panel p-2 min-w-[200px] z-50"
    transitionscale={{ duration: 200, start: 0.95 }}
  >
    <button
      use:melt={$item}
      onclick={() => onView(caseData.id)}
      class="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-nier-white/50 dark:hover:bg-nier-black/50 nier-transition w-full text-left"
    >
      <Eye class="w-4 h-4 text-nier-gray dark:text-nier-silver" />
      <span>View Details</span>
    </button>

    <button
      use:melt={$item}
      onclick={() => onEdit(caseData.id)}
      class="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-nier-white/50 dark:hover:bg-nier-black/50 nier-transition w-full text-left"
    >
      <Edit class="w-4 h-4 text-nier-gray dark:text-nier-silver" />
      <span>Edit Case</span>
    </button>

    <div use:melt={$separator} class="h-px bg-nier-light-gray dark:bg-nier-gray/30 my-2"></div>

    <button
      use:melt={$item}
      onclick={() => onArchive(caseData.id)}
      class="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-nier-amber/10 text-nier-amber nier-transition w-full text-left"
    >
      <Archive class="w-4 h-4" />
      <span>Archive</span>
    </button>

    <button
      use:melt={$item}
      onclick={() => onDelete(caseData.id)}
      class="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-harvard-crimson/10 text-harvard-crimson nier-transition w-full text-left"
    >
      <Trash2 class="w-4 h-4" />
      <span>Delete</span>
    </button>
  </div>
{/if}

<style lang="css">
  /* @unocss-include */
  /* Add smooth line clamp transitions */
  .line-clamp-1 {
    display: -webkit-box;
    -webkit-line-clamp: 1;
    line-clamp: 1;
    -webkit-box-orient: vertical
    overflow: hidden
  }

  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical
    overflow: hidden
  }
</style>

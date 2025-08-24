

<script lang="ts">
  import { fly, scale } from 'svelte/transition';
  import { createDropdownMenu, melt } from '@melt-ui/svelte';
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
  } from 'lucide-svelte';

  interface CaseData {
    id: string;
    status: keyof typeof statusConfig;
    priority: keyof typeof priorityConfig;
    title?: string;
    description?: string;
    updatedAt?: string | Date;
    created: string | Date;
    stats: {
      documents: number;
      evidence: number;
      witnesses: number;
    };
    progress?: number;
    tags?: string[];
    assignee?: {
      name: string;
      avatar?: string;
    };
  }

  interface Props {
    caseData: CaseData;
    onView?: (id: string) => void;
    onEdit?: (id: string) => void;
    onArchive?: (id: string) => void;
    onDelete?: (id: string) => void;
  }

  let {
    caseData,
    onView = () => {},
    onEdit = () => {},
    onArchive = () => {},
    onDelete = () => {}
  }: Props = $props();

  // Melt UI dropdown menu for actions
  const {
    elements: { trigger, menu, item, separator },
    states: { open }
  } = createDropdownMenu({
    forceVisible: true,
  });

  // Status configurations
  const statusConfig = {
    active: {
      icon: CheckCircle,
      class: 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400',
      label: 'Active'
    },
    pending: {
      icon: Clock,
      class: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400',
      label: 'Pending'
    },
    closed: {
      icon: CheckCircle,
      class: 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400',
      label: 'Closed'
    },
    archived: {
      icon: Archive,
      class: 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300',
      label: 'Archived'
    }
  }

  // Priority configurations
  const priorityConfig = {
    critical: {
      class: 'border-l-4 border-red-500',
      icon: '🚨',
      color: 'text-red-600 dark:text-red-400'
    },
    high: {
      class: 'border-l-4 border-orange-500',
      icon: '⚠️',
      color: 'text-orange-600 dark:text-orange-400'
    },
    medium: {
      class: 'border-l-4 border-yellow-500',
      icon: '📌',
      color: 'text-yellow-600 dark:text-yellow-400'
    },
    low: {
      class: 'border-l-4 border-green-500',
      icon: '📍',
      color: 'text-green-600 dark:text-green-400'
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
  class="case-card {currentPriority.class} group relative overflow-hidden bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 shadow-sm hover:shadow-md transition-all duration-200"
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
          <span class="text-sm font-mono text-gray-600 dark:text-gray-300">
            {caseData.id}
          </span>
          <span class="{currentPriority.color} text-lg" title="{caseData.priority} priority">
            {currentPriority.icon}
          </span>
        </div>

        <h3 class="text-lg font-semibold text-gray-900 dark:text-white line-clamp-1 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-200">
          {caseData.title}
        </h3>

        {#if caseData.description}
          <p class="text-sm text-gray-600 dark:text-gray-300 line-clamp-2 mt-1">
            {caseData.description}
          </p>
        {/if}
      </div>

      <div class="flex items-center gap-2">
        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {currentStatus.class}">
          {#key currentStatus.icon}
            <svelte:component this={currentStatus.icon} class="w-3 h-3 mr-1" />
          {/key}
          {currentStatus.label}
        </span>

        <button
          use:melt={$trigger}
          class="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-200"
          aria-label="More options"
        >
          <MoreVertical class="w-5 h-5 text-gray-600 dark:text-gray-300" />
        </button>
      </div>
    </div>

    <!-- Stats Grid -->
    <div class="grid grid-cols-3 gap-4 mb-4">
      <div class="text-center p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
        <div class="flex items-center justify-center gap-1 mb-1">
          <FileText class="w-4 h-4 text-gray-500 dark:text-gray-400" />
          <p class="text-xl font-bold text-blue-600 dark:text-blue-400">
            {caseData.stats.documents}
          </p>
        </div>
        <p class="text-xs text-gray-600 dark:text-gray-300">Documents</p>
      </div>

      <div class="text-center p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
        <div class="flex items-center justify-center gap-1 mb-1">
          <AlertTriangle class="w-4 h-4 text-gray-500 dark:text-gray-400" />
          <p class="text-xl font-bold text-blue-600 dark:text-blue-400">
            {caseData.stats.evidence}
          </p>
        </div>
        <p class="text-xs text-gray-600 dark:text-gray-300">Evidence</p>
      </div>

      <div class="text-center p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
        <div class="flex items-center justify-center gap-1 mb-1">
          <Users class="w-4 h-4 text-gray-500 dark:text-gray-400" />
          <p class="text-xl font-bold text-blue-600 dark:text-blue-400">
            {caseData.stats.witnesses}
          </p>
        </div>
        <p class="text-xs text-gray-600 dark:text-gray-300">Witnesses</p>
      </div>
    </div>

    <!-- Progress Bar (if applicable) -->
    {#if caseData.progress !== undefined}
      <div class="mb-4">
        <div class="flex justify-between items-center mb-1">
          <span class="text-xs text-gray-600 dark:text-gray-300">Progress</span>
          <span class="text-xs font-medium text-gray-900 dark:text-white">{caseData.progress}%</span>
        </div>
        <div class="h-2 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
          <div
            class="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-300"
            style="width: {caseData.progress}%"
          ></div>
        </div>
      </div>
    {/if}

    <!-- Tags -->
    {#if caseData.tags && caseData.tags.length > 0}
      <div class="flex flex-wrap gap-2 mb-4">
        {#each caseData.tags as tag}
          <span class="text-xs px-2 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
            #{tag}
          </span>
        {/each}
      </div>
    {/if}

    <!-- Footer -->
    <div class="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-600">
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
              <div class="w-6 h-6 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
                <span class="text-xs font-bold text-white">
                  {caseData.assignee.name.charAt(0).toUpperCase()}
                </span>
              </div>
            {/if}
            <span class="text-sm text-gray-600 dark:text-gray-300">
              {caseData.assignee.name}
            </span>
          </div>
        {/if}
      </div>

      <div class="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
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
        class="flex-1 bg-blue-600 text-white hover:bg-blue-700 px-4 py-2 rounded-md transition-colors duration-200 text-sm font-medium flex items-center justify-center gap-2"
      >
        <Eye class="w-4 h-4" />
        View Details
      </button>
      <button
        onclick={() => onEdit(caseData.id)}
        class="bg-gray-200 text-gray-800 hover:bg-gray-300 px-4 py-2 rounded-md transition-colors duration-200 text-sm font-medium flex items-center justify-center"
      >
        <Edit class="w-4 h-4" />
      </button>
    </div>
  </div>

</div>

<!-- Dropdown Menu -->
{#if $open}
  <div
    use:melt={$menu}
    class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-2 min-w-[200px] z-50"
    transition:scale={{ duration: 200, start: 0.95 }}
  >
    <button
      use:melt={$item}
      onclick={() => onView(caseData.id)}
      class="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 w-full text-left"
    >
      <Eye class="w-4 h-4 text-gray-600 dark:text-gray-300" />
      <span>View Details</span>
    </button>

    <button
      use:melt={$item}
      onclick={() => onEdit(caseData.id)}
      class="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 w-full text-left"
    >
      <Edit class="w-4 h-4 text-gray-600 dark:text-gray-300" />
      <span>Edit Case</span>
    </button>

    <div use:melt={$separator} class="h-px bg-gray-200 dark:bg-gray-600 my-2"></div>

    <button
      use:melt={$item}
      onclick={() => onArchive(caseData.id)}
      class="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-amber-50 dark:hover:bg-amber-900/20 text-amber-600 dark:text-amber-400 transition-colors duration-200 w-full text-left"
    >
      <Archive class="w-4 h-4" />
      <span>Archive</span>
    </button>

    <button
      use:melt={$item}
      onclick={() => onDelete(caseData.id)}
      class="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-red-50 dark:hover:bg-red-900/20 text-red-600 dark:text-red-400 transition-colors duration-200 w-full text-left"
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
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>


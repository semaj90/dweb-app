<script lang="ts">
  import { Badge } from "$lib/components/ui/index";
  import type { Case } from "$lib/types/api";
  import { formatDistanceToNow } from "date-fns";
  import {
    Archive,
    Calendar,
    CheckCircle,
    Clock,
    FileText,
    User,
  } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";

  export let caseData: Case;
  export let isActive = false;
  export let disabled = false;

  const dispatch = createEventDispatcher();

  function handleClick() {
    if (!disabled) {
      dispatch("click");
    }
  }

  function handleStatusChange(event: Event) {
    event.stopPropagation();
    const target = event.target as HTMLSelectElement;
    dispatch("statusChange", target.value);
  }

  function getStatusColor(status: string) {
    switch (status) {
      case "open":
        return "bg-green-100 text-green-800";
      case "in_progress":
        return "bg-yellow-100 text-yellow-800";
      case "closed":
        return "bg-blue-100 text-blue-800";
      case "archived":
        return "bg-gray-100 text-gray-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  }

  function getPriorityColor(priority: string) {
    switch (priority) {
      case "low":
        return "bg-green-100 text-green-800";
      case "medium":
        return "bg-yellow-100 text-yellow-800";
      case "high":
        return "bg-orange-100 text-orange-800";
      case "urgent":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case "open":
        return CheckCircle;
      case "in_progress":
        return Clock;
      case "closed":
        return Archive;
      case "archived":
        return Archive;
      default:
        return FileText;
    }
  }

  $: statusIcon = getStatusIcon(caseData.status);
  $: formattedDate = formatDistanceToNow(new Date(caseData.openedAt), {
    addSuffix: true,
  });
</script>

<div
  class="mx-auto px-4 max-w-7xl"
  class:active={isActive}
  class:disabled
  onclick={handleClick}
  onkeydown={(e) => e.key === "Enter" && handleClick()}
  role="button"
  tabindex={0}
>
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      <!-- Case Title and Number -->
      <div class="mx-auto px-4 max-w-7xl">
        <svelte:component
          this={statusIcon}
          class="mx-auto px-4 max-w-7xl"
        />
        <h3 class="mx-auto px-4 max-w-7xl">
          {caseData.title}
        </h3>
      </div>

      <!-- Case Number -->
      <p class="mx-auto px-4 max-w-7xl">
        Case #{caseData.caseNumber}
      </p>

      <!-- Status and Priority Badges -->
      <div class="mx-auto px-4 max-w-7xl">
        <Badge variant="outline" class={getStatusColor(caseData.status)}>
          {caseData.status.replace("_", " ")}
        </Badge>
        <Badge variant="outline" class={getPriorityColor(caseData.priority)}>
          {caseData.priority}
        </Badge>
      </div>

      <!-- Metadata -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <Calendar class="mx-auto px-4 max-w-7xl" />
          {formattedDate}
        </div>
        {#if caseData.defendantName}
          <div class="mx-auto px-4 max-w-7xl">
            <User class="mx-auto px-4 max-w-7xl" />
            {caseData.defendantName}
          </div>
        {/if}
        {#if caseData.evidenceCount > 0}
          <div class="mx-auto px-4 max-w-7xl">
            <FileText class="mx-auto px-4 max-w-7xl" />
            {caseData.evidenceCount} evidence
          </div>
        {/if}
      </div>

      <!-- Court Date if available -->
      {#if caseData.courtDate}
        <div class="mx-auto px-4 max-w-7xl">
          <Calendar class="mx-auto px-4 max-w-7xl" />
          Court: {new Date(caseData.courtDate).toLocaleDateString()}
        </div>
      {/if}
    </div>

    <!-- Quick Actions -->
    <div class="mx-auto px-4 max-w-7xl">
      <select
        class="mx-auto px-4 max-w-7xl"
        value={caseData.status}
        onchange={handleStatusChange}
        onclick={(e) => e.stopPropagation()}
      >
        <option value="open">Open</option>
        <option value="in_progress">In Progress</option>
        <option value="closed">Closed</option>
        <option value="archived">Archived</option>
      </select>
    </div>
  </div>
</div>

<style>
  .case-list-item.active {
    background-color: #dbeafe;
    border-left: 4px solid #3b82f6;
  }

  .case-list-item.disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .case-list-item:hover:not(.disabled) {
    background-color: #f9fafb;
  }
</style>

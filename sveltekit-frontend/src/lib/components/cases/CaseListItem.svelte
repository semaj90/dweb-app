<script lang="ts">
  import { Badge } from "$lib/components/ui/index";
  import type { Case } from '$lib/types';
  import { formatDistanceToNow } from "date-fns";
  import { Archive, Calendar, CheckCircle, Clock, FileText, User as UserIcon } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";

import type { Case } from '$lib/types';

  export let caseData: Case
  export let isActive = false;
  export let disabled = false;

  const dispatch = createEventDispatcher();

  function handleClick() {
    if (!disabled) {
      dispatch("click");
}}
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
}}
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
}}
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
}}
  $: statusIcon = getStatusIcon(caseData.status);
  $: formattedDate = formatDistanceToNow(new Date(caseData.openedAt), {
    addSuffix: true,
  });
</script>

<div
  class="container mx-auto px-4"
  class:active={isActive}
  class:disabled
  on:click={handleClick}
  on:keydown={(e) => e.key === "Enter" && handleClick()}
  role="button"
  tabindex={0}
>
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <!-- Case Title and Number -->
      <div class="container mx-auto px-4">
        <svelte:component
          this={statusIcon}
          class="container mx-auto px-4"
        />
        <h3 class="container mx-auto px-4">
          {caseData.title}
        </h3>
      </div>

      <!-- Case Number -->
      <p class="container mx-auto px-4">
        Case #{caseData.caseNumber}
      </p>

      <!-- Status and Priority Badges -->
      <div class="container mx-auto px-4">
        <Badge variant="outline">
          <span class={getStatusColor(caseData.status)}>{caseData.status.replace("_", " ")}</span>
        </Badge>
        <Badge variant="outline">
          <span class={getPriorityColor(caseData.priority)}>{caseData.priority}</span>
        </Badge>
      </div>

      <!-- Metadata -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <Calendar class="container mx-auto px-4" />
          {formattedDate}
        </div>
        {#if caseData.defendantName}
          <div class="container mx-auto px-4">
            <UserIcon class="container mx-auto px-4" />
            {caseData.defendantName}
          </div>
        {/if}
        {#if caseData.evidenceCount > 0}
          <div class="container mx-auto px-4">
            <FileText class="container mx-auto px-4" />
            {caseData.evidenceCount} evidence
          </div>
        {/if}
      </div>

      <!-- Court Date if available -->
      {#if caseData.courtDate}
        <div class="container mx-auto px-4">
          <Calendar class="container mx-auto px-4" />
          Court: {new Date(caseData.courtDate).toLocaleDateString()}
        </div>
      {/if}
    </div>

    <!-- Quick Actions -->
    <div class="container mx-auto px-4">
      <select
        class="container mx-auto px-4"
        value={caseData.status}
        on:change={handleStatusChange}
        on:click={(e) => e.stopPropagation()}
      >
        <option value="open">Open</option>
        <option value="in_progress">In Progress</option>
        <option value="closed">Closed</option>
        <option value="archived">Archived</option>
      </select>
    </div>
  </div>
</div>



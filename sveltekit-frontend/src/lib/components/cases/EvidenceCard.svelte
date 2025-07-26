<script lang="ts">
  import { formatDistanceToNow } from "date-fns";
  import {
    Archive,
    Calendar,
    Download,
    Edit,
    Eye,
    FileText,
    Headphones,
    Image,
    Trash2,
    Video,
  } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";

  export let evidence: any;
  export let disabled = false;

  const dispatch = createEventDispatcher();

  function getEvidenceIcon(type: string) {
    switch (type) {
      case "document":
        return FileText;
      case "photo":
        return Image;
      case "video":
        return Video;
      case "audio":
        return Headphones;
      case "physical":
        return Archive;
      case "digital":
        return FileText;
      case "testimony":
        return FileText;
      default:
        return FileText;
}}
  function getTypeColor(type: string) {
    switch (type) {
      case "document":
        return "bg-blue-100 text-blue-800";
      case "photo":
        return "bg-purple-100 text-purple-800";
      case "video":
        return "bg-red-100 text-red-800";
      case "audio":
        return "bg-green-100 text-green-800";
      case "physical":
        return "bg-yellow-100 text-yellow-800";
      case "digital":
        return "bg-indigo-100 text-indigo-800";
      case "testimony":
        return "bg-orange-100 text-orange-800";
      default:
        return "bg-gray-100 text-gray-800";
}}
  $: evidenceIcon = getEvidenceIcon(evidence.evidenceType || evidence.type);
  $: formattedDate = formatDistanceToNow(new Date(${1} || new Date()), {
    addSuffix: true,
  });

  function handleEdit() {
    if (!disabled) {
      dispatch("edit", evidence);
}}
  function handleDelete() {
    if (!disabled) {
      dispatch("delete", evidence);
}}
  function handleView() {
    if (!disabled) {
      dispatch("view", evidence);
}}
  function handleDownload() {
    if (!disabled) {
      dispatch("download", evidence);
}}
</script>

<div
  class="container mx-auto px-4"
  class:disabled
>
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <svelte:component this={evidenceIcon} class="container mx-auto px-4" />
      </div>

      <div class="container mx-auto px-4">
        <h4 class="container mx-auto px-4">
          {evidence.title}
        </h4>

        <div class="container mx-auto px-4">
          <span
            class="container mx-auto px-4"
          >
            {evidence.evidenceType || evidence.type}
          </span>
        </div>

        {#if evidence.description}
          <p class="container mx-auto px-4">
            {evidence.description}
          </p>
        {/if}

        <div class="container mx-auto px-4">
          <Calendar class="container mx-auto px-4" />
          {formattedDate}
        </div>
      </div>
    </div>

    <!-- Actions -->
    <div class="container mx-auto px-4">
      <button
        on:click={() => handleView()}
        class="container mx-auto px-4"
        title="View evidence"
        {disabled}
      >
        <Eye class="container mx-auto px-4" />
      </button>

      <button
        on:click={() => handleEdit()}
        class="container mx-auto px-4"
        title="Edit evidence"
        {disabled}
      >
        <Edit class="container mx-auto px-4" />
      </button>

      <button
        on:click={() => handleDownload()}
        class="container mx-auto px-4"
        title="Download evidence"
        {disabled}
      >
        <Download class="container mx-auto px-4" />
      </button>

      <button
        on:click={() => handleDelete()}
        class="container mx-auto px-4"
        title="Delete evidence"
        {disabled}
      >
        <Trash2 class="container mx-auto px-4" />
      </button>
    </div>
  </div>
</div>

<style>
  /* @unocss-include */
  .disabled {
    opacity: 0.6;
    pointer-events: none;
}
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
</style>

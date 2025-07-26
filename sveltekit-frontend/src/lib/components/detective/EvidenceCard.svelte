<script lang="ts">
  import type { Evidence } from "$lib/types/index";
  import { createEventDispatcher } from "svelte";

  export let item: Evidence;

  const dispatch = createEventDispatcher();

  function getEvidenceIcon(type: string) {
    switch (type) {
      case "document":
        return "i-lucide-file-text";
      case "image":
        return "i-lucide-image";
      case "video":
        return "i-lucide-video";
      case "audio":
        return "i-lucide-mic";
      case "digital":
        return "i-lucide-hard-drive";
      default:
        return "i-lucide-file";
}}
  function getTypeColor(type: string) {
    switch (type) {
      case "document":
        return "bg-blue-50 text-blue-700";
      case "image":
        return "bg-green-50 text-green-700";
      case "video":
        return "bg-purple-50 text-purple-700";
      case "audio":
        return "bg-red-50 text-red-700";
      case "digital":
        return "bg-orange-50 text-orange-700";
      default:
        return "bg-gray-50 text-gray-700";
}}
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}
  function formatDate(date: string | Date): string {
    return new Date(date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
}
</script>

<div
  class="container mx-auto px-4"
  role="article"
  aria-label={item.title}
>
  <!-- Header -->
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div
        class="container mx-auto px-4"
      >
        <i
          class="container mx-auto px-4"
          aria-hidden="true"
        ></i>
      </div>
      <div class="container mx-auto px-4">
        <h3 class="container mx-auto px-4">{item.title}</h3>
        <p class="container mx-auto px-4">{item.fileName}</p>
      </div>
    </div>
    <!-- Quick Actions -->
    <div
      class="container mx-auto px-4"
    >
      <button
        class="container mx-auto px-4"
        aria-label="View Evidence"
        tabindex={0}
      >
        <i class="container mx-auto px-4" aria-hidden="true"></i>
      </button>
      <button
        class="container mx-auto px-4"
        aria-label="More Options"
        tabindex={0}
      >
        <i class="container mx-auto px-4" aria-hidden="true"></i>
      </button>
    </div>
  </div>
  <!-- Preview/Thumbnail -->
  {#if item.thumbnailUrl}
    <div class="container mx-auto px-4">
      <img
        src={item.thumbnailUrl}
        alt="Evidence preview"
        class="container mx-auto px-4"
        loading="lazy"
      />
    </div>
  {:else if item.evidenceType === "document"}
    <div
      class="container mx-auto px-4"
    >
      <div class="container mx-auto px-4">
        <i
          class="container mx-auto px-4"
          aria-hidden="true"
        ></i>
        <p class="container mx-auto px-4">Document</p>
      </div>
    </div>
  {:else}
    <div
      class="container mx-auto px-4"
    >
      <div class="container mx-auto px-4">
        <i
          class="container mx-auto px-4"
          aria-hidden="true"
        ></i>
        <p class="container mx-auto px-4">{item.evidenceType}</p>
      </div>
    </div>
  {/if}
  <!-- AI Summary Preview -->
  {#if item.aiSummary}
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <i class="container mx-auto px-4" aria-hidden="true"></i>
        <span class="container mx-auto px-4">AI Summary</span>
      </div>
      <p class="container mx-auto px-4">{item.aiSummary}</p>
    </div>
  {/if}
  <!-- Metadata -->
  <div class="container mx-auto px-4">
    <!-- Tags -->
    {#if item.tags && item.tags.length > 0}
      <div class="container mx-auto px-4">
        {#each item.tags.slice(0, 3) as tag}
          <span
            class="container mx-auto px-4"
          >
            {tag}
          </span>
        {/each}
        {#if item.tags.length > 3}
          <span
            class="container mx-auto px-4"
          >
            +{item.tags.length - 3}
          </span>
        {/if}
      </div>
    {/if}
    <!-- File Info -->
    <div class="container mx-auto px-4">
      <span>{formatFileSize(item.fileSize || 0)}</span>
      <span>{formatDate(item.createdAt)}</span>
    </div>
    <!-- Hash Verification -->
    {#if item.hash}
      <div class="container mx-auto px-4">
        <i
          class="container mx-auto px-4"
          aria-hidden="true"
        ></i>
        <span class="container mx-auto px-4">Verified</span>
      </div>
    {/if}
  </div>
</div>

<style>
  /* @unocss-include */
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    line-clamp: 2;
    overflow: hidden;
}
  .evidence-card:hover .group-hover\:opacity-100 {
    opacity: 1;
}
</style>

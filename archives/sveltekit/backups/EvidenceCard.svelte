<script lang="ts">
  import { createTooltip, melt } from "@melt-ui/svelte";
  import {
    Download,
    Edit3,
    Eye,
    FileText,
    Headphones,
    Image,
    Link,
    Tag,
    Trash2,
    Video,
  } from "lucide-svelte";
  import { quintOut } from "svelte/easing";
  import { fly, scale } from "svelte/transition";
  import type { Evidence } from "../../../lib/stores/report";

  export let evidence: Evidence;
  export let onView: (evidence: Evidence) => void = () => {};
  export let onEdit: (evidence: Evidence) => void = () => {};
  export let onDelete: (evidence: Evidence) => void = () => {};
  export let onDownload: (evidence: Evidence) => void = () => {};
  export let draggable = true;
  export let compact = false;
  export let expandOnHover = false;
  export const showPreview = true;

  let className = "";
  export { className as class };

  // Tooltip for metadata
  const {
    elements: { trigger: tooltipTrigger, content: tooltipContent },
    states: { open: tooltipOpen },
  } = createTooltip({
    positioning: { placement: "top" },
    openDelay: 500,
    closeDelay: 0,
  });

  // Get icon for evidence type
  const getIcon = (type: Evidence["type"]) => {
    switch (type) {
      case "document":
        return FileText;
      case "image":
        return Image;
      case "video":
        return Video;
      case "audio":
        return Headphones;
      case "link":
        return Link;
      default:
        return FileText;
    }
  };

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  // Get file size from metadata
  const fileSize = evidence.metadata?.size || 0;

  // Hover state for expansion
  let isHovered = false;

  $: IconComponent = getIcon(evidence.evidenceType || evidence.type);

  function handleMouseEnter() {
    if (expandOnHover) {
      isHovered = true;
    }
  }

  function handleMouseLeave() {
    if (expandOnHover) {
      isHovered = false;
    }
  }
</script>

<div
  class="mx-auto px-4 max-w-7xl"
  class:draggable
  class:expanded={isHovered}
  transition:scale={{ duration: 200, easing: quintOut "
  onmouseenter={handleMouseEnter}
  onmouseleave={handleMouseLeave}
  role="article"
>
  <!-- Header -->
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl" data-type={evidence.evidenceType || evidence.type}>
      <svelte:component this={IconComponent} size={16} />
      <span class="mx-auto px-4 max-w-7xl">{evidence.evidenceType || evidence.type}</span>
    </div>

    <div class="mx-auto px-4 max-w-7xl">
      <button
        class="mx-auto px-4 max-w-7xl"
        onclick={() => onView(evidence)}
        title="View evidence"
      >
        <Eye size={14} />
      </button>

      {#if evidence.url || evidence.file}
        <button
          class="mx-auto px-4 max-w-7xl"
          onclick={() => onDownload(evidence)}
          title="Download"
        >
          <Download size={14} />
        </button>
      {/if}

      <button
        class="mx-auto px-4 max-w-7xl"
        onclick={() => onEdit(evidence)}
        title="Edit evidence"
      >
        <Edit3 size={14} />
      </button>

      <button
        class="mx-auto px-4 max-w-7xl"
        onclick={() => onDelete(evidence)}
        title="Delete evidence"
      >
        <Trash2 size={14} />
      </button>
    </div>
  </div>

  <!-- Content -->
  <div class="mx-auto px-4 max-w-7xl">
    <!-- Preview (for images/videos) -->
    {#if evidence.evidenceType || evidence.type === "image" && evidence.url}
      <div class="mx-auto px-4 max-w-7xl">
        <img
          src={evidence.url}
          alt={evidence.title}
          loading="lazy"
          on:error={(e) => {
            const target = e.currentTarget as HTMLImageElement;
            target.style.display = "none";
          "
        />
      </div>
    {:else if evidence.evidenceType || evidence.type === "video" && evidence.url}
      <div class="mx-auto px-4 max-w-7xl">
        <video src={evidence.url} preload="metadata" controls={false} muted>
          <track kind="captions" />
        </video>
        <div class="mx-auto px-4 max-w-7xl">
          <Video size={24} />
        </div>
      </div>
    {/if}

    <!-- Title and Description -->
    <div class="mx-auto px-4 max-w-7xl">
      <h3 class="mx-auto px-4 max-w-7xl" use:melt={$tooltipTrigger}>
        {evidence.title}
      </h3>

      {#if evidence.description && !compact}
        <p class="mx-auto px-4 max-w-7xl">
          {evidence.description}
        </p>
      {/if}

      <!-- Metadata -->
      <div class="mx-auto px-4 max-w-7xl">
        {#if evidence.metadata?.createdAt}
          <span class="mx-auto px-4 max-w-7xl">
            {new Date(evidence.createdAt).toLocaleDateString()}
          </span>
        {/if}

        {#if fileSize > 0}
          <span class="mx-auto px-4 max-w-7xl">
            {formatFileSize(fileSize)}
          </span>
        {/if}

        {#if evidence.metadata?.format}
          <span class="mx-auto px-4 max-w-7xl">
            {evidence.metadata.format.toUpperCase()}
          </span>
        {/if}
      </div>

      <!-- Tags -->
      {#if evidence.tags && evidence.tags.length > 0}
        <div class="mx-auto px-4 max-w-7xl">
          {#each evidence.tags.slice(0, 3) as tag}
            <span class="mx-auto px-4 max-w-7xl">
              <Tag size={10} />
              {tag}
            </span>
          {/each}
          {#if evidence.tags.length > 3}
            <span class="mx-auto px-4 max-w-7xl">+{evidence.tags.length - 3}</span>
          {/if}
        </div>
      {/if}
    </div>
  </div>

  <!-- Footer (if has URL) -->
  {#if evidence.url && evidence.evidenceType || evidence.type === "link"}
    <div class="mx-auto px-4 max-w-7xl">
      <a
        href={evidence.url}
        target="_blank"
        rel="noopener noreferrer"
        class="mx-auto px-4 max-w-7xl"
      >
        <Link size={14} />
        Open Link
      </a>
    </div>
  {/if}
</div>

<!-- Tooltip -->
{#if $tooltipOpen}
  <div
    use:melt={$tooltipContent}
    class="mx-auto px-4 max-w-7xl"
    transition:fly={{ y: -5, duration: 150 "
  >
    <div class="mx-auto px-4 max-w-7xl">
      <strong>{evidence.title}</strong>
      {#if evidence.description}
        <p>{evidence.description}</p>
      {/if}
      {#if evidence.metadata}
        <div class="mx-auto px-4 max-w-7xl">
          {#each Object.entries(evidence.metadata) as [key, value]}
            <div><strong>{key}:</strong> {value}</div>
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .evidence-card {
    background: var(--pico-card-background-color, #ffffff);
    border: 1px solid var(--pico-border-color, #e2e8f0);
    border-radius: 0.75rem;
    overflow: hidden;
    transition: all 0.2s ease;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .evidence-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    border-color: var(--pico-primary, #3b82f6);
  }

  .evidence-card.draggable {
    cursor: grab;
  }

  .evidence-card.draggable:active {
    cursor: grabbing;
  }

  .evidence-card.compact {
    font-size: 0.875rem;
  }

  .evidence-card.expanded {
    transform: scale(1.05);
    z-index: 10;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
  }

  .evidence-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem;
    background: var(--pico-card-sectioning-background-color, #f8fafc);
    border-bottom: 1px solid var(--pico-border-color, #e2e8f0);
  }

  .evidence-type-badge {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.25rem 0.5rem;
    border-radius: 0.375rem;
    font-size: 0.75rem;
    font-weight: 500;
    border: 1px solid;
  }

  .type-label {
    text-transform: capitalize;
  }

  .evidence-actions {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    opacity: 0;
    transition: opacity 0.2s ease;
  }

  .evidence-card:hover .evidence-actions {
    opacity: 1;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 1.75rem;
    height: 1.75rem;
    border: none;
    background: none;
    border-radius: 0.25rem;
    color: var(--pico-muted-color, #6b7280);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .action-btn:hover {
    background: var(--pico-primary-background, #f3f4f6);
  }

  .action-btn.view:hover {
    color: var(--pico-primary, #3b82f6);
  }

  .action-btn.edit:hover {
    color: var(--pico-ins-color, #10b981);
  }

  .action-btn.delete:hover {
    color: var(--pico-del-color, #ef4444);
  }

  .action-btn.download:hover {
    color: var(--pico-primary, #6366f1);
  }

  .evidence-content {
    padding: 0.75rem;
  }

  .evidence-preview {
    position: relative;
    width: 100%;
    margin-bottom: 0.75rem;
    border-radius: 0.5rem;
    overflow: hidden;
    background: var(--pico-card-sectioning-background-color, #f8fafc);
  }

  .evidence-preview img,
  .evidence-preview video {
    width: 100%;
    height: auto;
    max-height: 200px;
    object-fit: cover;
  }

  .video-overlay {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0, 0, 0, 0.6);
    border-radius: 50%;
    padding: 0.75rem;
    color: white;
  }

  .evidence-info {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .evidence-title {
    font-size: 1rem;
    font-weight: 600;
    margin: 0;
    color: var(--pico-color, #111827);
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .evidence-description {
    font-size: 0.875rem;
    color: var(--pico-muted-color, #6b7280);
    line-height: 1.5;
    margin: 0.5rem 0;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .evidence-metadata {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.5rem 0;
  }

  .metadata-item {
    font-size: 0.75rem;
    color: var(--pico-muted-color, #9ca3af);
    background: var(--pico-card-sectioning-background-color, #f1f5f9);
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
  }

  .evidence-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    margin-top: 0.5rem;
  }

  .tag {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.75rem;
    background: var(--pico-primary-background, #eff6ff);
    color: var(--pico-primary, #3b82f6);
    padding: 0.125rem 0.375rem;
    border-radius: 0.375rem;
    border: 1px solid var(--pico-primary-border, #bfdbfe);
  }

  .tag-more {
    font-size: 0.75rem;
    color: var(--pico-muted-color, #6b7280);
    font-weight: 500;
  }

  .evidence-footer {
    padding: 0.75rem;
    border-top: 1px solid var(--pico-border-color, #e2e8f0);
    background: var(--pico-card-sectioning-background-color, #f8fafc);
  }

  .evidence-link {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    color: var(--pico-primary, #3b82f6);
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 500;
    transition: color 0.15s ease;
  }

  .evidence-link:hover {
    color: var(--pico-primary-hover, #2563eb);
  }

  .tooltip {
    background: var(--pico-card-background-color, #1f2937);
    color: white;
    padding: 0.75rem;
    border-radius: 0.5rem;
    font-size: 0.875rem;
    max-width: 16rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    z-index: 50;
  }

  .tooltip-content p {
    margin: 0.5rem 0;
    opacity: 0.9;
  }

  .tooltip-metadata {
    margin-top: 0.5rem;
    font-size: 0.75rem;
    opacity: 0.8;
  }

  .tooltip-metadata div {
    margin: 0.25rem 0;
  }

  .evidence-type-badge[data-type="document"] {
    color: #2563eb;
    background-color: #eff6ff;
    border-color: #bfdbfe;
  }

  .evidence-type-badge[data-type="image"] {
    color: #16a34a;
    background-color: #f0fdf4;
    border-color: #bbf7d0;
  }

  .evidence-type-badge[data-type="video"] {
    color: #9333ea;
    background-color: #faf5ff;
    border-color: #e9d5ff;
  }

  .evidence-type-badge[data-type="audio"] {
    color: #ea580c;
    background-color: #fff7ed;
    border-color: #fed7aa;
  }

  .evidence-type-badge[data-type="link"] {
    color: #4f46e5;
    background-color: #eef2ff;
    border-color: #c7d2fe;
  }
</style>
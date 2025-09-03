<!-- EvidenceCard.svelte - Fixed for Svelte 5 -->
<script lang="ts">
  import Badge from "$lib/components/ui/Badge.svelte";
  import Button from "$lib/components/ui/button/Button.svelte";
  import { Card, CardContent, CardHeader } from "$lib/components/ui/card";
  import type { Evidence } from "$lib/types/index";

  // --- SVELTE 5 PROPS ---
  // The new way to define props using runes with callback functions
  interface Props {
    item: Evidence;
    onView?: (item: Evidence) => void;
    onMoreOptions?: (item: Evidence) => void;
  }
  
  let { item, onView, onMoreOptions }: Props = $props();

  // --- Helper Functions ---
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
    }
  }

  function getTypeColor(type: string) {
    switch (type) {
      case "document":
        return "bg-blue-50 text-blue-700 dark:bg-blue-950 dark:text-blue-300";
      case "image":
        return "bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-300";
      case "video":
        return "bg-purple-50 text-purple-700 dark:bg-purple-950 dark:text-purple-300";
      case "audio":
        return "bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300";
      case "digital":
        return "bg-orange-50 text-orange-700 dark:bg-orange-950 dark:text-orange-300";
      default:
        return "bg-gray-50 text-gray-700 dark:bg-gray-800 dark:text-gray-300";
    }
  }

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

<Card
  class="group hover:shadow-md transition-shadow duration-200 cursor-pointer"
  role="article"
  aria-label={item.title}
>
  <CardHeader class="pb-3">
    <div class="flex items-start justify-between">
      <div class="flex items-center gap-3">
        <div
          class="flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center {getTypeColor(
            item.evidenceType || item.type || 'document'
          )}"
        >
          <i
            class="{getEvidenceIcon(item.evidenceType || item.type || 'document')} w-5 h-5"
            aria-hidden="true"
          ></i>
        </div>
        <div class="min-w-0 flex-1">
          <h3 class="font-semibold text-sm text-foreground truncate">
            {item.title}
          </h3>
          <p class="text-xs text-muted-foreground truncate">
            {item.fileName || "No filename"}
          </p>
        </div>
      </div>

      <!-- Quick Actions -->
      <div
        class="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity"
      >
        <Button
          variant="ghost"
          size="sm"
          class="h-8 w-8 p-0"
          aria-label="View Evidence"
          on:click={() => onView?.(item)}
        >
          <i class="i-lucide-eye w-4 h-4" aria-hidden="true"></i>
        </Button>
        <Button
          variant="ghost"
          size="sm"
          class="h-8 w-8 p-0"
          aria-label="More Options"
          on:click={() => onMoreOptions?.(item)}
        >
          <i class="i-lucide-more-horizontal w-4 h-4" aria-hidden="true"></i>
        </Button>
      </div>
    </div>
  </CardHeader>

  <CardContent class="space-y-3">
    <!-- Preview/Thumbnail -->
    {#if item.thumbnailUrl}
      <div class="aspect-video bg-muted rounded-md overflow-hidden">
        <img
          src={item.thumbnailUrl}
          alt="Evidence preview"
          class="w-full h-full object-cover"
          loading="lazy"
        />
      </div>
    {:else}
      <div
        class="aspect-video bg-muted bg-opacity-30 rounded-md flex items-center justify-center border-2 border-dashed border-muted-foreground border-opacity-25"
      >
        <div class="text-center">
          <i
            class="{getEvidenceIcon(
              item.evidenceType || item.type || 'document'
            )} w-8 h-8 mx-auto mb-2 text-muted-foreground"
            aria-hidden="true"
          ></i>
          <p class="text-xs text-muted-foreground capitalize">
            {item.evidenceType || item.type || 'document'}
          </p>
        </div>
      </div>
    {/if}

    <!-- AI Summary Preview -->
    {#if item.aiSummary || item.analysis?.aiSummary}
      <div class="bg-muted/50 rounded-md p-3 space-y-2">
        <div class="flex items-center gap-2">
          <i class="i-lucide-brain w-4 h-4 text-primary" aria-hidden="true"></i>
          <span class="text-xs font-medium text-primary">AI Summary</span>
        </div>
        <p class="text-xs text-muted-foreground line-clamp-2">
          {item.aiSummary || item.analysis?.aiSummary}
        </p>
      </div>
    {/if}

    <!-- Metadata -->
    <div class="space-y-3">
      <!-- Tags -->
      {#if item.tags && item.tags.length > 0}
        <div class="flex flex-wrap gap-1">
          {#each item.tags.slice(0, 3) as tag}
            <Badge variant="secondary" class="text-xs px-2 py-0.5">
              {tag}
            </Badge>
          {/each}
          {#if item.tags.length > 3}
            <Badge variant="outline" class="text-xs px-2 py-0.5">
              +{item.tags.length - 3}
            </Badge>
          {/if}
        </div>
      {/if}

      <!-- File Info -->
      <div
        class="flex items-center justify-between text-xs text-muted-foreground"
      >
        <span>{formatFileSize(item.fileSize || 0)}</span>
        <span>{formatDate(item.createdAt || item.timeline?.createdAt || new Date())}</span>
      </div>

      <!-- Hash Verification -->
      {#if item.hash}
        <div class="flex items-center gap-2">
          <i
            class="i-lucide-shield-check w-4 h-4 text-green-600"
            aria-hidden="true"
          ></i>
          <span class="text-xs text-green-600 font-medium">Verified</span>
        </div>
      {/if}
    </div>
  </CardContent>
</Card>

<style>
  /* @unocss-include */
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>
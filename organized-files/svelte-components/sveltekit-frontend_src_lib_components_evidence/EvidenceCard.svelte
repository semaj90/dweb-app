<script lang="ts">
  interface Props { evidence: ExtendedEvidence; onView: (evidence: Evidence) => void;,
    onEdit: (evidence: Evidence) => void;,
    onDelete: (evidence: Evidence) => void;,
    onDownload: (evidence: Evidence) => void;,
    draggable?: any;
    compact?: any;
    expandOnHover?: any;
   }
  let { evidence,
    onView = > void = () => { },
    onEdit = > void = () => {},
    onDelete = > void = () => {},
    onDownload = > void = () => {},
    draggable = true,
    compact = false,
    expandOnHover = false
  } = $props();



  
  import { Download,
    PenLine,
    Eye,
    FileText,
    Headphones,
    Image,
    Link,
    Tag,
    Trash2,
    Video,
   } from "lucide-svelte";
  import { quintOut  } from "svelte/easing";
  import { fly, scale  } from "svelte/transition";
  import type { Evidence  } from '$lib/stores/report';

  type ExtendedEvidence = Evidence & { evidenceType?: string;
    fileSize?: number;
    createdAt?: Date | string;
    updatedAt?: Date | string;
   };

                  export const showPreview = true;

  const { elements: { trigger:, tooltipTrigger, content: tooltipContent },
    states: { open: tooltipOpen },
  } = createTooltip({ positioning: {, placement: "top"  },
    openDelay: 500,
    closeDelay: 0,
  });

  const getIcon = (type: Evidence["type"]) => { switch (type) {
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
        return, FileText;
     }
  };

  const formatFileSize = (bytes: number): string => { if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
   };

  const fileSize = evidence.metadata?.size || evidence.fileSize || 0;
  let isHovered = false;

  let IconComponent = $derived(getIcon()
    ["document", "image", "video", "audio", "link"].includes(evidence.evidenceType || evidence.type)
      ? (evidence.evidenceType || evidence.type) as Evidence["type"]
      : "document"
  );

  function handleMouseEnter() { if (expandOnHover) {
      isHovered = true;
     }
  }
  function handleMouseLeave() { if (expandOnHover) {
      isHovered = false;
     }
  }
</script>

<div
  class="bg-white border border-gray-200 rounded-xl overflow-hidden transition-all duration-200 shadow hover:shadow-lg hover:border-blue-500
    relative
    { compact ? 'text-sm' : '' }
    { draggable ? 'cursor-grab active:cursor-grabbing' : '' }
    { isHovered ? 'scale-105 z-10 shadow-2xl' : '' }"
  transitionscale={ { duration: 200, easing: quintOut }}
  onmouseenter={ handleMouseEnter }
  onmouseleave={ handleMouseLeave }
  role="article"
>
  <!-- Header -->
  <div class="flex items-center justify-between px-3 py-3 bg-gray-50 border-b border-gray-200">
    <div class="flex items-center gap-1 px-2 py-1 rounded text-xs font-medium border capitalize"
      class:bg-blue-50={ evidence.evidenceType === 'document' || evidence.type === 'document' }
      class:text-blue-700={ evidence.evidenceType === 'document' || evidence.type === 'document' }
      class:border-blue-200={ evidence.evidenceType === 'document' || evidence.type === 'document' }
      class:bg-green-50={ evidence.evidenceType === 'image' || evidence.type === 'image' }
      class:text-green-700={ evidence.evidenceType === 'image' || evidence.type === 'image' }
      class:border-green-200={ evidence.evidenceType === 'image' || evidence.type === 'image' }
      class:bg-purple-50={ evidence.evidenceType === 'video' || evidence.type === 'video' }
      class:text-purple-700={ evidence.evidenceType === 'video' || evidence.type === 'video' }
      class:border-purple-200={ evidence.evidenceType === 'video' || evidence.type === 'video' }
      class:bg-orange-50={ evidence.evidenceType === 'audio' || evidence.type === 'audio' }
      class:text-orange-700={ evidence.evidenceType === 'audio' || evidence.type === 'audio' }
      class:border-orange-200={ evidence.evidenceType === 'audio' || evidence.type === 'audio' }
      class:bg-indigo-50={ evidence.evidenceType === 'link' || evidence.type === 'link' }
      class:text-indigo-700={ evidence.evidenceType === 'link' || evidence.type === 'link' }
      class:border-indigo-200={ evidence.evidenceType === 'link' || evidence.type === 'link' }
      data-type={ evidence.evidenceType || evidence.type }
    >
      <svelte:component this={ IconComponent } size={ 16 } />
      <span>{ evidence.evidenceType || evidence.type }</span>
    </div>

    <div class="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
      <button
        class="flex items-center justify-center w-7 h-7 rounded text-gray-500 hover:bg-gray-100 hover:text-blue-600"
        onclick={ () => onView(evidence as Evidence) }
        title="View evidence"
      >
        <Eye size={ 14 } />
      </button>

      { #if evidence.url || evidence.file }
        <button
          class="flex items-center justify-center w-7 h-7 rounded text-gray-500 hover:bg-gray-100 hover:text-indigo-600"
          onclick={ () => onDownload(evidence as Evidence) }
          title="Download"
        >
          <Download size={ 14 } />
        </button>
      { /if }

      <button
        class="flex items-center justify-center w-7 h-7 rounded text-gray-500 hover:bg-gray-100 hover:text-green-600"
        onclick={ () => onEdit(evidence as Evidence) }
        title="Edit evidence"
      >
        <PenLine size={ 14 } />
      </button>

      <button
        class="flex items-center justify-center w-7 h-7 rounded text-gray-500 hover:bg-gray-100 hover:text-red-600"
        onclick={ () => onDelete(evidence as Evidence) }
        title="Delete evidence"
      >
        <Trash2 size={ 14 } />
      </button>
    </div>
  </div>

  <!-- Content -->
  <div class="px-3 py-3">
    <!-- Preview (for images/videos) -->
    { #if (evidence.evidenceType || evidence.type) === "image" && evidence.url }
      <div class="relative w-full mb-3 rounded-lg overflow-hidden bg-gray-50">
        <img
          src={ evidence.url }
          alt={ evidence.title }
          loading="lazy"
          class="w-full h-auto max-h-48 object-cover"
          onerror={ (e) => {
            const target = e.currentTarget as HTMLImageElement;
            target.style.display = "none";
           }}
        />
      </div>
    { :else if (evidence.evidenceType || evidence.type) === "video" && evidence.url }
      <div class="relative w-full mb-3 rounded-lg overflow-hidden bg-gray-50">
        <video src={ evidence.url } preload="metadata" controls={ false } muted class="w-full h-auto max-h-48 object-cover">
          <track kind="captions" />
        </video>
        <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-black/60 rounded-full p-3 text-white">
          <Video size={ 24 } />
        </div>
      </div>
    { /if }

    <!-- Title and Description -->
    <div class="flex flex-col gap-2">
      <h3 class="font-semibold text-base text-gray-900 leading-tight line-clamp-2" use:melt={ $tooltipTrigger }>
        { evidence.title }
      </h3>

      { #if evidence.description && !compact }
        <p class="text-sm text-gray-500 leading-snug line-clamp-3">
          { evidence.description }
        </p>
      { /if }

      <!-- Metadata -->
      <div class="flex flex-wrap gap-2 my-2">
        { #if evidence.metadata?.createdAt || evidence.createdAt }
          <span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded">
            { new Date(evidence.metadata?.createdAt || evidence.createdAt || '').toLocaleDateString() }
          </span>
        { /if }

        { #if fileSize > 0 }
          <span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded">
            { formatFileSize(fileSize) }
          </span>
        { /if }

        { #if evidence.metadata?.format }
          <span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded">
            { evidence.metadata.format.toUpperCase() }
          </span>
        { /if }
      </div>

      <!-- Tags -->
      { #if evidence.tags && evidence.tags.length > 0 }
        <div class="flex flex-wrap gap-1 mt-2">
          { #each evidence.tags.slice(0, 3) as tag }
            <span class="flex items-center gap-1 text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded border border-blue-200">
              <Tag size={ 10 } />
              { tag }
            </span>
          { /each }
          { #if evidence.tags.length > 3 }
            <span class="text-xs text-gray-500 font-medium">+{ evidence.tags.length - 3 }</span>
          { /if }
        </div>
      { /if }
    </div>
  </div>

  <!-- Footer (if has URL) -->
  { #if evidence.url && ((evidence.evidenceType || evidence.type) === "link") }
    <div class="px-3 py-3 border-t border-gray-200 bg-gray-50">
      <a
        href={ evidence.url }
        target="_blank"
        rel="noopener noreferrer"
        class="flex items-center gap-1 text-indigo-600 hover:text-indigo-800 text-sm font-medium"
      >
        <Link size={ 14 } />
        Open Link
      </a>
    </div>
  { /if }
</div>

<!-- Tooltip -->
{ #if $tooltipOpen }
  <div
    use:melt={ $tooltipContent }
    class="mx-auto px-4"
    transitionfly={ { y: -5, duration: 150  }}
  >
    <div class="mx-auto px-4">
      <strong>{ evidence.title }</strong>
      { #if evidence.description }
        <p>{ evidence.description }</p>
      { /if }
      { #if evidence.metadata }
        <div class="mx-auto px-4">
          { #each Object.entries(evidence.metadata) as [key, value] }
            <div><strong>{ key }:</strong> { value }</div>
          { /each }
        </div>
      { /if }
    </div>
  </div>
{ /if }

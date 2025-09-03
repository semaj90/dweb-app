<script lang="ts">
  import EvidenceFilesManager from '$lib/components/evidence/EvidenceFilesManager.svelte';
</script>

<section class="container mx-auto py-8 space-y-6">
  <header class="space-y-2">
    <h1 class="text-2xl font-bold">Evidence Management</h1>
    <p class="text-sm text-muted-foreground max-w-prose">Upload, view, and manage evidence files stored in MinIO and indexed in PostgreSQL. Future enhancements (embeddings, semantic search, download links, AI summaries) will appear here.</p>
  </header>

  <EvidenceFilesManager />
</section>

<style>
  .container { max-width: 1200px; }
</style>

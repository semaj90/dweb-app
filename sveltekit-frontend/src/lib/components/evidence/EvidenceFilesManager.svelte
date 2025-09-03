<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';

  interface EvidenceItem {
    id: number;
    title: string;
    evidence_type: string;
    file_size: number;
    mime_type: string;
    uploaded_at: string;
  }

  let items: EvidenceItem[] = [];
  let loading = false;
  let error: string | null = null;
  let file: File | null = null;
  let title = '';
  let evidenceType = 'UNKNOWN';
  let uploading = false;
  let refreshInterval: any;

  async function fetchItems() {
    loading = true; error = null;
    try {
      const res = await fetch('/api/evidence-files?limit=100');
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Failed to load');
      items = data.items;
    } catch (e: any) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  function formatSize(bytes: number) {
    if (bytes === 0) return '0 B';
    const k = 1024; const sizes = ['B','KB','MB','GB'];
    const i = Math.floor(Math.log(bytes)/Math.log(k));
    return (bytes/Math.pow(k,i)).toFixed(2)+' '+sizes[i];
  }

  async function handleUpload(e: Event) {
    e.preventDefault();
    if (!file) { error = 'Select a file'; return; }
    uploading = true; error = null;
    try {
      const fd = new FormData();
      fd.append('file', file);
      if (title) fd.append('title', title);
      if (evidenceType) fd.append('evidence_type', evidenceType);
      const res = await fetch('/api/evidence-files', { method: 'POST', body: fd });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Upload failed');
      file = null; title = ''; evidenceType = 'UNKNOWN';
      await fetchItems();
    } catch (err: any) {
      error = err.message;
    } finally {
      uploading = false;
    }
  }

  async function deleteItem(id: number) {
    if (!confirm('Delete this file?')) return;
    try {
      const res = await fetch(`/api/evidence-files?id=${id}`, { method: 'DELETE' });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Delete failed');
      items = items.filter(i => i.id !== id);
    } catch (err: any) {
      error = err.message;
    }
  }

  async function downloadItem(id: number) {
    try {
      const res = await fetch(`/api/evidence-files?download=${id}`);
      const data = await res.json();
      if (!data.success || !data.url) throw new Error(data.error || 'Download link failed');
      // Open in new tab (preserves original name if browser uses query content-disposition)
      window.open(data.url, '_blank');
    } catch (err: any) {
      error = err.message;
    }
  }

  onMount(() => {
    fetchItems();
    // Optional polling (disable if not needed)
    // refreshInterval = setInterval(fetchItems, 30000);
    return () => refreshInterval && clearInterval(refreshInterval);
  });
</script>

<div class="evidence-manager space-y-6">
  <div class="upload-panel p-4 border rounded-md bg-background/50 backdrop-blur">
    <h2 class="font-semibold mb-2">Upload Evidence File</h2>
    <form on:submit|preventDefault={handleUpload} class="space-y-3">
      <div>
        <Label class="text-xs uppercase tracking-wide">Title</Label>
        <Input placeholder="Optional title" bind:value={title} />
      </div>
      <div>
        <Label class="text-xs uppercase tracking-wide">Type</Label>
        <select bind:value={evidenceType} class="border rounded px-2 py-1 text-sm w-full bg-background">
          <option value="UNKNOWN">Auto</option>
          <option value="PDF">PDF</option>
          <option value="IMAGE">Image</option>
          <option value="VIDEO">Video</option>
          <option value="AUDIO">Audio</option>
          <option value="TEXT">Text</option>
        </select>
      </div>
      <div>
        <Label class="text-xs uppercase tracking-wide">File</Label>
        <input type="file" class="block w-full text-sm" on:change={(e:any)=> file = e.currentTarget.files?.[0] || null} />
      </div>
      <div class="flex gap-2">
        <Button type="submit" disabled={uploading || !file}>{uploading ? 'Uploading...' : 'Upload'}</Button>
        <Button type="button" variant="secondary" on:click={()=>{ file=null; title=''; evidenceType='UNKNOWN'; }}>Reset</Button>
      </div>
    </form>
    {#if uploading}
      <p class="text-xs mt-2 text-muted-foreground">Uploading...</p>
    {/if}
  </div>

  <div class="list-panel p-4 border rounded-md bg-background/50 backdrop-blur">
    <div class="flex items-center justify-between mb-3">
      <h2 class="font-semibold">Stored Evidence ({items.length})</h2>
      <Button size="sm" variant="outline" on:click={fetchItems} disabled={loading}>{loading ? 'Loading...' : 'Refresh'}</Button>
    </div>
    {#if error}
      <div class="text-sm text-destructive mb-2">{error}</div>
    {/if}
    {#if loading && items.length === 0}
      <p class="text-sm text-muted-foreground">Loading...</p>
    {:else if items.length === 0}
      <p class="text-sm text-muted-foreground">No files uploaded yet.</p>
    {:else}
      <table class="w-full text-sm border-collapse">
        <thead>
          <tr class="text-left border-b">
            <th class="py-1 pr-2">Title</th>
            <th class="py-1 pr-2">Type</th>
            <th class="py-1 pr-2">Size</th>
            <th class="py-1 pr-2">Uploaded</th>
            <th class="py-1 pr-2" colspan="2"></th>
          </tr>
        </thead>
        <tbody>
          {#each items as item}
            <tr class="border-b hover:bg-accent/30">
              <td class="py-1 pr-2 max-w-[220px] truncate" title={item.title}>{item.title}</td>
              <td class="py-1 pr-2 uppercase text-xs">{item.evidence_type}</td>
              <td class="py-1 pr-2 tabular-nums">{formatSize(item.file_size)}</td>
              <td class="py-1 pr-2 text-xs">{new Date(item.uploaded_at).toLocaleString()}</td>
              <td class="py-1 pr-2 text-right space-x-1 whitespace-nowrap">
                <Button size="sm" variant="outline" on:click={() => downloadItem(item.id)}>Download</Button>
                <Button size="sm" variant="destructive" on:click={() => deleteItem(item.id)}>Delete</Button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>
</div>

<style>
  .evidence-manager { width: 100%; }
  table th { font-weight: 600; font-size: 0.75rem; letter-spacing: .05em; }
</style>

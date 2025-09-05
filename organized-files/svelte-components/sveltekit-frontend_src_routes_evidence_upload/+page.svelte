<script lang="ts">
  import FileUploadForm from '$lib/components/upload/FileUploadForm.svelte';
  import { page } from '$app/stores';
  import type { PageData } from './$types';

  let { data } = $props<{ data: PageData }>();
  
  // Get caseId from URL params if present
  const caseId = $derived($page.url.searchParams.get('caseId') || '');
</script>

<div class="container mx-auto py-8">
  <FileUploadForm {data} {caseId} />
</div>
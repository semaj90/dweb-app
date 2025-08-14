<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  
  let hashInput = '81d9c48f998f9025eb8f72e28a6c4f921ed407dd75891a9e9a8778c9ad5711bd';
  let searchResult: any = null;
  let loading = false;
  let error = '';

  onMount(() => {
    // Check if hash was provided in URL
    const urlHash = $page.url.searchParams.get('hash');
    if (urlHash) {
      hashInput = urlHash;
      searchByHash();
}
  });

  async function searchByHash() {
    if (!hashInput || hashInput.length !== 64) {
      error = 'Please enter a valid 64-character SHA256 hash';
      return;
}
    loading = true;
    error = '';
    searchResult = null;

    try {
      const response = await fetch(`/api/evidence/hash?hash=${hashInput}`);
      const result = await response.json();
      
      if (response.ok) {
        searchResult = result;
      } else {
        error = result.error || 'Search failed';
}
    } catch (e) {
      error = 'Network error occurred';
    } finally {
      loading = false;
}}
  async function verifyIntegrity(evidenceId: string) {
    if (!evidenceId) return;
    
    loading = true;
    error = '';

    try {
      const response = await fetch('/api/evidence/hash', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hash: hashInput, evidenceId })
      });

      const result = await response.json();
      
      if (response.ok) {
        alert(`Integrity Check: ${result.message}`);
      } else {
        error = result.error || 'Verification failed';
}
    } catch (e) {
      error = 'Network error occurred';
    } finally {
      loading = false;
}}
  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text).then(() => {
      alert('Copied to clipboard!');
    });
}
</script>

<svelte:head>
  <title>Evidence Hash Verification - Legal Case Management</title>
</svelte:head>

<div class="space-y-4">
  <div class="space-y-4">
    <h1 class="space-y-4">🔐 Evidence Hash Verification</h1>
    <p class="space-y-4">
      Verify file integrity and search for evidence using SHA256 hashes
    </p>
  </div>

  <div class="space-y-4">
    <div class="space-y-4">
      <h2 class="space-y-4">Hash Search & Verification</h2>
      
      <div class="space-y-4">
        <label for="hash-input" class="space-y-4">
          SHA256 Hash (64 characters)
        </label>
        <div class="space-y-4">
          <input
            id="hash-input"
            type="text"
            bind:value={hashInput}
            placeholder="Enter SHA256 hash..."
            class="space-y-4"
            maxlength="64"
          />
          <button 
            onclick={() => searchByHash()} 
            disabled={loading || !hashInput}
            class="space-y-4"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
        <p class="space-y-4">
          Example: 81d9c48f998f9025eb8f72e28a6c4f921ed407dd75891a9e9a8778c9ad5711bd
        </p>
      </div>

      {#if error}
        <div class="space-y-4">
          <strong>Error:</strong> {error}
        </div>
      {/if}

      {#if searchResult}
        <div class="space-y-4">
          <h3 class="space-y-4">Search Results</h3>
          
          {#if searchResult.found}
            <div class="space-y-4">
              <strong>✅ {searchResult.message}</strong>
            </div>
            
            <div class="space-y-4">
              {#each searchResult.evidence as item}
                <div class="space-y-4">
                  <div class="space-y-4">
                    <div class="space-y-4">
                      <h4 class="space-y-4">{item.title}</h4>
                      <span class="space-y-4">
                        ID: {item.id}
                      </span>
                    </div>
                    
                    <div class="space-y-4">
                      <div>
                        <p><strong>File:</strong> {item.fileName || 'N/A'}</p>
                        <p><strong>Size:</strong> {item.fileSize ? (item.fileSize / 1024).toFixed(1) + ' KB' : 'N/A'}</p>
                        <p><strong>Type:</strong> {item.fileType || 'N/A'}</p>
                      </div>
                      <div>
                        <p><strong>Case:</strong> {item.caseName || 'N/A'} ({item.caseNumber || 'N/A'})</p>
                        <p><strong>Uploaded by:</strong> {item.uploaderName || 'N/A'}</p>
                        <p><strong>Uploaded:</strong> {item.uploadedAt ? new Date(item.uploadedAt).toLocaleString() : 'N/A'}</p>
                      </div>
                    </div>
                    
                    <div class="space-y-4">
                      <strong>Hash:</strong> {item.hash}
                      <button 
                        onclick={() => copyToClipboard(item.hash)}
                        class="space-y-4"
                        title="Copy hash"
                      >
                        📋
                      </button>
                    </div>
                    
                    {#if item.description}
                      <p class="space-y-4">{item.description}</p>
                    {/if}
                    
                    <div class="space-y-4">
                      <button 
                        onclick={() => verifyIntegrity(item.id)}
                        disabled={loading}
                        class="space-y-4"
                      >
                        Verify Integrity
                      </button>
                      
                      {#if item.fileUrl}
                        <a 
                          href={item.fileUrl}
                          target="_blank"
                          class="space-y-4"
                        >
                          View File
                        </a>
                      {/if}
                    </div>
                  </div>
                </div>
              {/each}
            </div>
          {:else}
            <div class="space-y-4">
              <strong>⚠️ {searchResult.message}</strong>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  </div>

  <div class="space-y-4">
    <div class="space-y-4">
      <h2 class="space-y-4">About Hash Verification</h2>
      
      <div class="space-y-4">
        <p>
          This tool allows you to search for evidence files by their SHA256 hash and verify file integrity.
        </p>
        
        <h3>How it works:</h3>
        <ul>
          <li><strong>File Upload:</strong> When evidence is uploaded, a SHA256 hash is automatically calculated and stored</li>
          <li><strong>Hash Search:</strong> Search for evidence using the exact 64-character SHA256 hash</li>
          <li><strong>Integrity Verification:</strong> Compare provided hashes with stored hashes to detect file tampering</li>
        </ul>
        
        <h3>Use cases:</h3>
        <ul>
          <li>Verify that an evidence file hasn't been modified</li>
          <li>Find evidence files by their cryptographic fingerprint</li>
          <li>Ensure chain of custody integrity</li>
          <li>Cross-reference files across different cases</li>
        </ul>
        
        <div class="space-y-4">
          <p class="space-y-4">
            <strong>Security Note:</strong> SHA256 hashes provide cryptographic assurance that files have not been altered.
            Each file has a unique hash that changes if even a single byte is modified.
          </p>
        </div>
      </div>
    </div>
  </div>
</div>

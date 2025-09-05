

  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { Button } from 'bits-ui';
  import { Card } from 'bits-ui';
  import { Tabs } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { Separator } from 'bits-ui';
  import DocumentUploader from '$lib/components/DocumentUploader.svelte';
  import VectorSearchInterface from '$lib/components/VectorSearchInterface.svelte';
  import AISummarization from '$lib/components/AISummarization.svelte';
  import { fly, fade } from 'svelte/transition';

  // State
  let activeTab = $state('upload');
  let selectedDocument = $state(null);
  let recentUploads = $state([]);
  let searchResults = $state([]);
  let workspaceStats = $state({
    totalDocuments: 0,
    totalProcessed: 0,
    totalSearches: 0,
    avgProcessingTime: 0
  });

  // Reactive derived values
  let hasDocuments = $derived(workspaceStats.totalDocuments > 0);
  let hasSearchResults = $derived(searchResults.length > 0);

  /**
   * Handle document upload completion
   */
  function handleUploadComplete(event: CustomEvent) {
    const { fileId, result } = event.detail;
    
    if (result.status === 'completed' && result.result) {
      // Add to recent uploads
      recentUploads = [
        {
          id: fileId,
          filename: result.result.document.fileName || 'Unknown',
          status: 'completed',
          uploadedAt: new Date(),
          documentId: result.result.documentId
        },
        ...recentUploads.slice(0, 9) // Keep last 10
      ];

      // Update stats
      workspaceStats.totalDocuments++;
      workspaceStats.totalProcessed++;
      workspaceStats.avgProcessingTime = (
        (workspaceStats.avgProcessingTime * (workspaceStats.totalProcessed - 1) + result.metadata.processingTime) / 
        workspaceStats.totalProcessed
      );

      // Auto-switch to search tab after successful upload
      setTimeout(() => {
        activeTab = 'search';
      }, 2000);
    }
  }

  /**
   * Handle search completion
   */
  function handleSearchComplete(event: CustomEvent<{ 
    query: string; 
    results: Array 
  }>) {
    const { results } = event.detail;
    searchResults = results;
    workspaceStats.totalSearches++;
  }

  /**
   * Handle document selection from search
   */
  function handleDocumentSelect(event: CustomEvent) {
    const { document } = event.detail;
    selectedDocument = document;
    activeTab = 'analyze';
  }

  /**
   * Load recent document for analysis
   */
  async function loadDocumentForAnalysis(documentId: string) {
    try {
      const response = await fetch(`/api/documents/${documentId}`);
      if (response.ok) {
        const document = await response.json();
        selectedDocument = document;
        activeTab = 'analyze';
      }
    } catch (error) {
      console.error('Failed to load document:', error);
    }
  }

  /**
   * Clear selected document
   */
  function clearSelectedDocument() {
    selectedDocument = null;
  }

  /**
   * Format processing time
   */
  function formatProcessingTime(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  }

  /**
   * Format file size
   */
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Get status color
   */
  function getStatusColor(status: string): string {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'processing': return 'bg-blue-100 text-blue-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }



  AI Legal Workspace - Document Analysis & Search
  



  
  
    
      
        
          
            
              
                
              
            
            AI Legal Workspace
          
        

        
        
          
            
              
            
            {workspaceStats.totalDocuments} documents
          
          
            
              
            
            {workspaceStats.totalSearches} searches
          
          {#if workspaceStats.totalProcessed > 0}
            
              
                
              
              avg {formatProcessingTime(workspaceStats.avgProcessingTime)}
            
          {/if}
        
      
    
  

  
  
    
      
      
        
          
            
          
          Upload
        
        
        
          
            
          
          Search
        
        
        
          
            
          
          Analyze
        
        
        
          
            
          
          History
        
      

      
      
        
          
            
              Upload Legal Documents
            
            
              Upload PDF, DOCX, or text files for AI-powered analysis including summarization, 
              entity extraction, risk assessment, and semantic search indexing.
            
          

          <DocumentUploader
            accept=".pdf,.docx,.txt,.json"
            maxSize={50 * 1024 * 1024}
            multiple={true}
            on:complete={handleUploadComplete}
            class="max-w-4xl mx-auto"
          />

          
          {#if recentUploads.length > 0}
            
              
                Recent Uploads
                
                  {#each recentUploads.slice(0, 5) as upload}
                    
                      
                        
                          
                        
                        
                          {upload.filename}
                          
                            {upload.uploadedAt.toLocaleTimeString()}
                          
                        
                      
                      
                        
                          {upload.status}
                        
                        {#if upload.documentId}
                          <Button.Root
                            size="sm"
                            variant="outline"
                            on:click={() => loadDocumentForAnalysis(upload.documentId!)}
                          >
                            Analyze
                          
                        {/if}
                      
                    
                  {/each}
                
              
            
          {/if}
        
      

      
      
        
          
            
              Semantic Document Search
            
            
              Search through your legal documents using AI-powered semantic understanding. 
              Find relevant documents even when they don't contain exact keyword matches.
            
          

          <VectorSearchInterface
            on:search={handleSearchComplete}
            on:select={handleDocumentSelect}
            maxResults={20}
            showFilters={true}
          />
        
      

      
      
        
          
            
              AI Document Analysis
            
            
              Get comprehensive AI-powered analysis including summaries, entity extraction, 
              risk assessment, and custom insights for your legal documents.
            
          

          {#if selectedDocument}
            
              
              
                
                  
                    
                      {selectedDocument.title}
                      
                        Type: {selectedDocument.documentType}
                        {#if selectedDocument.practiceArea}
                          •
                          Area: {selectedDocument.practiceArea.replace('_', ' ')}
                        {/if}
                        •
                        Jurisdiction: {selectedDocument.jurisdiction}
                        {#if selectedDocument.fileSize}
                          •
                          Size: {formatFileSize(selectedDocument.fileSize)}
                        {/if}
                      
                    
                    <Button.Root
                      variant="outline"
                      on:click={clearSelectedDocument}
                    >
                      Close Analysis
                    
                  
                
              

              
              <AISummarization
                documentContent={selectedDocument.content}
                documentTitle={selectedDocument.title}
                documentId={selectedDocument.id}
                autoSummarize={true}
                showAnalysisTools={true}
                class="max-w-5xl mx-auto"
              />
            
          {:else}
            
            
              
                
                  
                
              
              Select a document to analyze
              
                Choose a document from your search results or recent uploads to begin AI analysis
              
              
                <Button.Root
                  on:click={() => activeTab = 'search'}
                  disabled={!hasDocuments}
                >
                  Search Documents
                
                <Button.Root
                  variant="outline"
                  on:click={() => activeTab = 'upload'}
                >
                  Upload New Document
                
              
            
          {/if}
        
      

      
      
        
          
            
              Workspace History
            
            
              Review your document uploads, searches, and analysis history.
            
          

          
            
            
              
                Recent Uploads
                {#if recentUploads.length > 0}
                  
                    {#each recentUploads as upload}
                      
                        
                          {upload.filename}
                          
                            {upload.uploadedAt.toLocaleString()}
                          
                        
                        
                          
                            {upload.status}
                          
                          {#if upload.documentId}
                            <Button.Root
                              size="sm"
                              variant="outline"
                              on:click={() => loadDocumentForAnalysis(upload.documentId!)}
                            >
                              View
                            
                          {/if}
                        
                      
                    {/each}
                  
                {:else}
                  No uploads yet
                {/if}
              
            

            
            
              
                Recent Search Results
                {#if hasSearchResults}
                  
                    {#each searchResults.slice(0, 5) as result}
                      
                        
                          
                            {result.title}
                            
                              {Math.round(result.similarity * 100)}% similarity
                            
                          
                          <Button.Root
                            size="sm"
                            variant="outline"
                            on:click={() => handleDocumentSelect({ detail: { document: result } })}
                          >
                            Analyze
                          
                        
                      
                    {/each}
                  
                {:else}
                  No search results yet
                {/if}
              
            
          

          
          
            
              Workspace Statistics
              
                
                  {workspaceStats.totalDocuments}
                  Total Documents
                
                
                  {workspaceStats.totalProcessed}
                  Processed
                
                
                  {workspaceStats.totalSearches}
                  Searches
                
                
                  
                    {workspaceStats.totalProcessed > 0 ? formatProcessingTime(workspaceStats.avgProcessingTime) : '-'}
                  
                  Avg Process Time
                
              
            
          
        
      
    
  



  .ai-workspace {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

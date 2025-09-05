<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import { 
    generateMCPPrompt, 
    commonMCPQueries, 
    validateMCPRequest, 
    formatMCPResponse,
    type MCPToolRequest 
  } from '$lib/utils/mcp-helpers';

  // Component state
  let selectedTool = 'analyze-stack';
  let component = 'sveltekit';
  let context = 'legal-ai';
  let area = 'performance';
  let feature = '';
  let requirements = '';
  let library = '';
  let topic = '';
  
  // RAG-specific variables
  let ragQuery = '';
  let maxResults = 10;
  let confidenceThreshold = 0.7;
  let ragCaseId = '';
  let documentTypes = '';
  let filePath = '';
  let documentType = 'general';
  let documentTitle = '';
  let documentId = '';
  let integrationType = 'api-integration';
  
  let result = '';
  let loading = false;
  let error = '';

  // Available options
  const tools = [
    { value: 'analyze-stack', label: 'Analyze Stack Component' },
    { value: 'generate-best-practices', label: 'Generate Best Practices' },
    { value: 'suggest-integration', label: 'Suggest Integration' },
    { value: 'resolve-library-id', label: 'Resolve Library ID' },
    { value: 'get-library-docs', label: 'Get Library Documentation' },
    { value: 'rag-query', label: 'RAG Query Legal Documents' },
    { value: 'rag-upload-document', label: 'RAG Upload Document' },
    { value: 'rag-get-stats', label: 'RAG System Statistics' },
    { value: 'rag-analyze-relevance', label: 'RAG Analyze Document Relevance' },
    { value: 'rag-integration-guide', label: 'RAG Integration Guide' }
  ];

  const components = [
    'sveltekit', 'drizzle', 'unocss', 'bits-ui', 'xstate', 'fabric.js', 
    'typescript', 'postgresql', 'gemma3', 'autogen', 'crewai', 'vllm'
  ];

  const contexts = ['legal-ai', 'gaming-ui', 'performance'];
  const areas = ['performance', 'security', 'ui-ux'];

  // Common queries for quick testing
  const quickQueries = [
    { name: 'Analyze SvelteKit', query: commonMCPQueries.analyzeSvelteKit },
    { name: 'Analyze Drizzle ORM', query: commonMCPQueries.analyzeDrizzle },
    { name: 'Performance Best Practices', query: commonMCPQueries.performanceBestPractices },
    { name: 'Security Best Practices', query: commonMCPQueries.securityBestPractices },
    { name: 'AI Chat Integration', query: commonMCPQueries.aiChatIntegration },
    { name: 'Document Upload Integration', query: commonMCPQueries.documentUploadIntegration },
    { name: 'RAG System Stats', query: commonMCPQueries.ragStats },
    { name: 'RAG API Integration', query: commonMCPQueries.ragApiIntegration },
    { name: 'RAG Search UI', query: commonMCPQueries.ragSearchUI }
  ];

  // Build MCP request from form data
  function buildRequest(): MCPToolRequest {
    const request: MCPToolRequest = { tool: selectedTool as any };

    switch (selectedTool) {
      case 'analyze-stack':
        request.component = component;
        request.context = context as any;
        break;
      case 'generate-best-practices':
        request.area = area as any;
        break;
      case 'suggest-integration':
        request.feature = feature;
        request.requirements = requirements;
        break;
      case 'resolve-library-id':
      case 'get-library-docs':
        request.library = library;
        if (selectedTool === 'get-library-docs' && topic) {
          request.topic = topic;
        }
        break;
      case 'rag-query':
        request.query = ragQuery;
        request.maxResults = maxResults;
        request.confidenceThreshold = confidenceThreshold;
        if (ragCaseId) request.caseId = ragCaseId;
        if (documentTypes) request.documentTypes = documentTypes.split(',').map(t => t.trim());
        break;
      case 'rag-upload-document':
        request.filePath = filePath;
        if (ragCaseId) request.caseId = ragCaseId;
        request.documentType = documentType;
        if (documentTitle) request.title = documentTitle;
        break;
      case 'rag-get-stats':
        // No additional parameters needed
        break;
      case 'rag-analyze-relevance':
        request.query = ragQuery;
        request.documentId = documentId;
        break;
      case 'rag-integration-guide':
        request.integrationType = integrationType as any;
        break;
    }

    return request;
  }

  // Execute MCP tool (simulated)
  async function executeTool() {
    loading = true;
    error = '';
    result = '';

    try {
      const request = buildRequest();
      
      // Validate request
      const validation = validateMCPRequest(request);
      if (!validation.valid) {
        throw new Error(`Invalid request: ${validation.errors.join(', ')}`);
      }

      // Generate prompt
      const prompt = generateMCPPrompt(request);
      
      // In a real implementation, this would call the actual MCP server
      // For now, simulate the response
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate network delay
      
      const mockResponse = generateMockResponse(request);
      result = formatMCPResponse(mockResponse);

    } catch (err) {
      error = err instanceof Error ? err.message : 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }

  // Execute common query
  async function executeQuickQuery(queryFn: () => MCPToolRequest) {
    const request = queryFn();
    
    // Update form with quick query values
    selectedTool = request.tool;
    component = request.component || '';
    context = request.context || 'legal-ai';
    area = request.area || 'performance';
    feature = request.feature || '';
    requirements = request.requirements || '';
    library = request.library || '';
    topic = request.topic || '';
    
    // RAG-specific fields
    ragQuery = request.query || '';
    maxResults = request.maxResults || 10;
    confidenceThreshold = request.confidenceThreshold || 0.7;
    ragCaseId = request.caseId || '';
    documentTypes = request.documentTypes?.join(', ') || '';
    filePath = request.filePath || '';
    documentType = request.documentType || 'general';
    documentTitle = request.title || '';
    documentId = request.documentId || '';
    integrationType = request.integrationType || 'api-integration';

    await executeTool();
  }

  // Generate mock response for demonstration
  function generateMockResponse(request: MCPToolRequest) {
    switch (request.tool) {
      case 'analyze-stack':
        return {
          content: [{
            type: 'text',
            text: `# Stack Analysis: ${request.component} (${request.context})

## Recommended Patterns for Legal AI
- Use ${request.component} with legal data security best practices
- Implement proper authentication and authorization
- Follow legal compliance requirements (GDPR, HIPAA)
- Optimize for prosecutor workflow efficiency

## Integration Points
- Connect with case management database
- Integrate with evidence processing pipeline
- Support multi-agent AI workflows (Autogen + CrewAI)
- Enable real-time collaboration features

## Performance Considerations
- Optimize for large legal document processing
- Implement efficient caching strategies
- Support offline capabilities for field work
- Scale for multi-user prosecution teams

## Security Best Practices
- Encrypt sensitive legal data at rest and in transit
- Implement row-level security for case access
- Use audit logging for all legal document access
- Secure API endpoints with proper authentication`
          }]
        };

      case 'generate-best-practices':
        return {
          content: [{
            type: 'text',
            text: `# ${request.area?.toUpperCase()} Best Practices for Legal AI

## Key Recommendations
${request.area === 'performance' ? `
- Use server-side rendering for legal document pages
- Implement progressive enhancement for offline access
- Optimize database queries with proper indexing
- Use Ollama for high-throughput AI inference
- Cache frequently accessed legal precedents
- Implement efficient vector similarity searches
` : request.area === 'security' ? `
- Encrypt all legal documents at rest and in transit
- Implement row-level security (RLS) for case data
- Use secure authentication with multi-factor auth
- Audit all evidence access and modifications
- Validate and sanitize all evidence uploads
- Monitor for unusual access patterns
` : `
- Design case-centric navigation for prosecutors
- Implement quick evidence search and filtering
- Use progressive disclosure for complex legal data
- Provide clear AI confidence indicators
- Follow WCAG 2.1 AA accessibility standards
- Use responsive design for mobile field work
`}
## Implementation Guidelines
- Follow legal industry compliance standards
- Implement proper error handling and user feedback
- Use consistent design patterns throughout
- Test with actual legal professionals
- Document all security and compliance measures`
          }]
        };

      case 'suggest-integration':
        return {
          content: [{
            type: 'text',
            text: `# Integration Suggestion: ${request.feature}

## Recommended Approach
Based on your SvelteKit legal AI stack:

### File Structure
\`\`\`
src/
├── routes/api/${request.feature?.toLowerCase().replace(/\s+/g, '-')}/
│   └── +server.ts
├── lib/components/${request.feature}/
│   ├── ${request.feature}Component.svelte
│   └── index.ts
└── lib/stores/${request.feature}Store.ts
\`\`\`

### Database Schema
- Add tables for ${request.feature} data
- Implement proper relationships with cases/evidence
- Add indexes for query performance
- Consider audit trail requirements

### API Design
- RESTful endpoints following SvelteKit conventions
- Proper error handling and validation
- Authentication checks for legal data access
- Rate limiting for AI-powered features

### Frontend Components
- Use Bits UI components for accessibility
- Implement proper loading and error states
- Follow legal UI patterns and branding
- Ensure mobile responsiveness

### Requirements Analysis
${request.requirements || 'No specific requirements provided'}

### Security Considerations
- Input validation and sanitization
- Proper authentication for legal data access
- Audit logging for compliance requirements
- Data encryption for sensitive information`
          }]
        };

      case 'resolve-library-id':
        const libraryMap: Record<string, string> = {
          'sveltekit': 'sveltekit',
          'svelte': 'sveltekit',
          'drizzle': 'drizzle',
          'unocss': 'unocss',
          'bits-ui': 'bits-ui',
          'xstate': 'xstate',
          'fabric.js': 'fabric-js'
        };
        const resolved = libraryMap[request.library?.toLowerCase() || ''] || request.library;
        
        return {
          content: [{
            type: 'text',
            text: `# Library ID Resolution

Library: ${request.library}
Resolved ID: ${resolved}

Available documentation: ${Object.keys(libraryMap).join(', ')}`
          }]
        };

      case 'get-library-docs':
        return {
          content: [{
            type: 'text',
            text: `# ${request.library} Documentation: ${request.topic || 'overview'}

## Documentation Content
Detailed documentation for ${request.library} covering ${request.topic || 'general usage'} in the context of legal AI applications.

### Key Concepts
- Integration patterns with SvelteKit
- Legal data handling considerations
- Performance optimization techniques
- Security best practices

### Code Examples
\`\`\`typescript
// Example integration code for ${request.library}
import { ${request.library} } from '${request.library}';

// Legal AI specific configuration
const config = {
  security: 'high',
  auditLogging: true,
  encryption: 'AES-256'
};
\`\`\`

### Best Practices
- Follow legal compliance requirements
- Implement proper error handling
- Use TypeScript for type safety
- Test with legal professional workflows`
          }]
        };

      default:
        return {
          content: [{
            type: 'text',
            text: `Tool ${request.tool} not implemented in mock response.`
          }]
        };
    }
  }

  // Clear form
  function clearForm() {
    component = 'sveltekit';
    context = 'legal-ai';
    area = 'performance';
    feature = '';
    requirements = '';
    library = '';
    topic = '';
    
    // Reset RAG fields
    ragQuery = '';
    maxResults = 10;
    confidenceThreshold = 0.7;
    ragCaseId = '';
    documentTypes = '';
    filePath = '';
    documentType = 'general';
    documentTitle = '';
    documentId = '';
    integrationType = 'api-integration';
    
    result = '';
    error = '';
  }
</script>

<div class="space-y-6 p-6 max-w-4xl mx-auto">
  <div class="border-b border-gray-200 pb-4">
    <h2 class="text-2xl font-bold text-gray-900">Context7 MCP Tools Demo</h2>
    <p class="text-gray-600 mt-2">
      Test and explore Context7 MCP tools for stack analysis, best practices, and integration guidance.
    </p>
  </div>

  <!-- Quick Actions -->
  <div class="bg-blue-50 p-4 rounded-lg">
    <h3 class="text-lg font-semibold mb-3">Quick Actions</h3>
    <div class="grid grid-cols-2 md:grid-cols-3 gap-2">
      {#each quickQueries as query}
        <button
          type="button"
          onclick={() => executeQuickQuery(query.query)}
          disabled={loading}
          class="px-3 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {query.name}
        </button>
      {/each}
    </div>
  </div>

  <!-- Tool Configuration -->
  <div class="bg-white border border-gray-200 rounded-lg p-6">
    <h3 class="text-lg font-semibold mb-4">Tool Configuration</h3>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <!-- Tool Selection -->
      <div>
        <label for="tool" class="block text-sm font-medium mb-2">Tool</label>
        <select
          id="tool"
          bind:value={selectedTool}
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {#each tools as tool}
            <option value={tool.value}>{tool.label}</option>
          {/each}
        </select>
      </div>

      <!-- Tool-specific inputs -->
      {#if selectedTool === 'analyze-stack'}
        <div>
          <label for="component" class="block text-sm font-medium mb-2">Component</label>
          <select
            id="component"
            bind:value={component}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {#each components as comp}
              <option value={comp}>{comp}</option>
            {/each}
          </select>
        </div>
        
        <div>
          <label for="context" class="block text-sm font-medium mb-2">Context</label>
          <select
            id="context"
            bind:value={context}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {#each contexts as ctx}
              <option value={ctx}>{ctx}</option>
            {/each}
          </select>
        </div>
      
      {:else if selectedTool === 'generate-best-practices'}
        <div>
          <label for="area" class="block text-sm font-medium mb-2">Area</label>
          <select
            id="area"
            bind:value={area}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {#each areas as ar}
              <option value={ar}>{ar}</option>
            {/each}
          </select>
        </div>
      
      {:else if selectedTool === 'suggest-integration'}
        <div>
          <label for="feature" class="block text-sm font-medium mb-2">Feature</label>
          <input
            id="feature"
            type="text"
            bind:value={feature}
            placeholder="e.g., AI chat component"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <div class="md:col-span-2">
          <label for="requirements" class="block text-sm font-medium mb-2">Requirements (optional)</label>
          <textarea
            id="requirements"
            bind:value={requirements}
            placeholder="e.g., real-time messaging, legal compliance, audit trails"
            rows="3"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          ></textarea>
        </div>
      
      {:else if selectedTool === 'resolve-library-id' || selectedTool === 'get-library-docs'}
        <div>
          <label for="library" class="block text-sm font-medium mb-2">Library</label>
          <input
            id="library"
            type="text"
            bind:value={library}
            placeholder="e.g., sveltekit, drizzle, bits-ui"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        {#if selectedTool === 'get-library-docs'}
          <div>
            <label for="topic" class="block text-sm font-medium mb-2">Topic (optional)</label>
            <input
              id="topic"
              type="text"
              bind:value={topic}
              placeholder="e.g., routing, schema, dialog"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        {/if}
      
      {:else if selectedTool === 'rag-query'}
        <div class="md:col-span-2">
          <label for="ragQuery" class="block text-sm font-medium mb-2">Legal Query</label>
          <textarea
            id="ragQuery"
            bind:value={ragQuery}
            placeholder="e.g., contract liability clauses, criminal evidence standards, case precedents..."
            rows="3"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          ></textarea>
        </div>
        
        <div>
          <label for="maxResults" class="block text-sm font-medium mb-2">Max Results</label>
          <input
            id="maxResults"
            type="number"
            bind:value={maxResults}
            min="1"
            max="50"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <div>
          <label for="confidenceThreshold" class="block text-sm font-medium mb-2">Confidence Threshold</label>
          <input
            id="confidenceThreshold"
            type="range"
            bind:value={confidenceThreshold}
            min="0.1"
            max="1"
            step="0.1"
            class="w-full"
          />
          <span class="text-sm text-gray-600">{confidenceThreshold}</span>
        </div>
        
        <div>
          <label for="ragCaseId" class="block text-sm font-medium mb-2">Case ID (optional)</label>
          <input
            id="ragCaseId"
            type="text"
            bind:value={ragCaseId}
            placeholder="e.g., CASE-2024-001"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <div>
          <label for="documentTypes" class="block text-sm font-medium mb-2">Document Types (optional)</label>
          <input
            id="documentTypes"
            type="text"
            bind:value={documentTypes}
            placeholder="e.g., contract, evidence, case_law"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      
      {:else if selectedTool === 'rag-upload-document'}
        <div class="md:col-span-2">
          <label for="filePath" class="block text-sm font-medium mb-2">File Path</label>
          <input
            id="filePath"
            type="text"
            bind:value={filePath}
            placeholder="e.g., /path/to/legal-document.pdf"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <div>
          <label for="ragCaseId" class="block text-sm font-medium mb-2">Case ID (optional)</label>
          <input
            id="ragCaseId"
            type="text"
            bind:value={ragCaseId}
            placeholder="e.g., CASE-2024-001"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <div>
          <label for="documentType" class="block text-sm font-medium mb-2">Document Type</label>
          <select
            id="documentType"
            bind:value={documentType}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="general">General</option>
            <option value="contract">Contract</option>
            <option value="evidence">Evidence</option>
            <option value="case_law">Case Law</option>
            <option value="statute">Statute</option>
            <option value="regulation">Regulation</option>
          </select>
        </div>
        
        <div>
          <label for="documentTitle" class="block text-sm font-medium mb-2">Title (optional)</label>
          <input
            id="documentTitle"
            type="text"
            bind:value={documentTitle}
            placeholder="e.g., Employment Contract v2.1"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      
      {:else if selectedTool === 'rag-analyze-relevance'}
        <div>
          <label for="ragQuery" class="block text-sm font-medium mb-2">Query</label>
          <input
            id="ragQuery"
            type="text"
            bind:value={ragQuery}
            placeholder="e.g., liability clauses"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <div>
          <label for="documentId" class="block text-sm font-medium mb-2">Document ID</label>
          <input
            id="documentId"
            type="text"
            bind:value={documentId}
            placeholder="e.g., doc-uuid-1234"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      
      {:else if selectedTool === 'rag-integration-guide'}
        <div>
          <label for="integrationType" class="block text-sm font-medium mb-2">Integration Type</label>
          <select
            id="integrationType"
            bind:value={integrationType}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="api-integration">API Integration</option>
            <option value="component-integration">Component Integration</option>
            <option value="search-ui">Search UI</option>
            <option value="document-upload">Document Upload</option>
          </select>
        </div>
      {/if}
    </div>

    <!-- Action Buttons -->
    <div class="flex gap-3 mt-6">
      <button
        type="button"
        onclick={executeTool}
        disabled={loading}
        class="flex-1 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {#if loading}
          <span class="flex items-center justify-center gap-2">
            <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            Executing...
          </span>
        {:else}
          Execute Tool
        {/if}
      </button>
      <button
        type="button"
        onclick={clearForm}
        class="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
      >
        Clear
      </button>
    </div>
  </div>

  <!-- Results -->
  {#if error}
    <div class="bg-red-50 border border-red-200 rounded-lg p-4">
      <h3 class="text-lg font-semibold text-red-800 mb-2">Error</h3>
      <p class="text-red-700">{error}</p>
    </div>
  {/if}

  {#if result}
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <h3 class="text-lg font-semibold mb-4">Result</h3>
      <div class="prose max-w-none">
        <pre class="whitespace-pre-wrap text-sm text-gray-800 bg-gray-50 p-4 rounded-md overflow-x-auto">{result}</pre>
      </div>
    </div>
  {/if}

  <!-- Usage Instructions -->
  <div class="bg-gray-50 p-6 rounded-lg">
    <h3 class="text-lg font-semibold mb-3">Usage Instructions</h3>
    <div class="space-y-2 text-sm text-gray-700">
      <p><strong>In Claude Desktop:</strong> Use natural language prompts like "analyze sveltekit with context legal-ai" or "generate best practices for security"</p>
      <p><strong>In VS Code:</strong> Use the MCP tools via Claude Code integration for code analysis and suggestions</p>
      <p><strong>In SvelteKit:</strong> Import and use the MCP helper functions from <code>$lib/utils/mcp-helpers</code></p>
      <p><strong>Common Queries:</strong> Use the quick action buttons above for frequently needed analysis</p>
    </div>
  </div>
</div>
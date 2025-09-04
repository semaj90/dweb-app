<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { writable } from 'svelte/store';
  
  const dispatch = createEventDispatcher();
  
  export let selectedNode: unknown = null;
  export let readOnly = false;
  
  // Enhanced form fields with auto-population
  let formData = writable({
    // Basic fields
    title: '',
    description: '',
    tags: [] as string[],
    customTags: [] as string[],
    
    // Entity fields (auto-populated by AI)
    people: [] as string[],
    locations: [] as string[],
    organizations: [] as string[],
    dates: [] as string[],
    
    // Legal fields
    evidenceType: 'other',
    legalRelevance: 'medium',
    legalCategories: [] as string[],
    confidentialityLevel: 'internal',
    urgencyLevel: 'normal',
    
    // Analysis fields
    keyFacts: [] as string[],
    potentialWitnesses: [] as string[],
    relatedCases: [] as string[],
    statutes: [] as string[],
    monetaryAmounts: [] as string[],
    actions: [] as string[],
    
    // Quality metrics
    qualityScore: 0,
    extractionConfidence: {
      people: 0,
      locations: 0,
      dates: 0,
      organizations: 0
    } as Record<string, number>,
    
    // Warnings and recommendations
    redFlags: [] as string[],
    recommendations: [] as string[]
  });
  
  // Form state
  let isLoading = false;
  let isSaving = false;
  let hasUnsavedChanges = false;
  let lastSavedAt: Date | null = null;
  let autoSaveTimer: ReturnType<typeof setTimeout>;
  
  // Custom input fields
  let customTag = '';
  let customPerson = '';
  let customLocation = '';
  let customOrganization = '';
  let customAction = '';
  
  // Evidence type options with icons
  const evidenceTypes = [
    { value: 'document', label: 'Document', icon: '📄' },
    { value: 'photo', label: 'Photo', icon: '🖼️' },
    { value: 'video', label: 'Video', icon: '🎥' },
    { value: 'audio', label: 'Audio', icon: '🎵' },
    { value: 'physical', label: 'Physical Evidence', icon: '🔍' },
    { value: 'digital', label: 'Digital Evidence', icon: '💾' },
    { value: 'testimony', label: 'Testimony', icon: '🗣️' },
    { value: 'other', label: 'Other', icon: '📁' }
  ];
  
  // Legal relevance options with colors
  const relevanceOptions = [
    { value: 'critical', label: 'Critical', color: 'bg-red-100 text-red-800 border-red-200' },
    { value: 'high', label: 'High', color: 'bg-orange-100 text-orange-800 border-orange-200' },
    { value: 'medium', label: 'Medium', color: 'bg-yellow-100 text-yellow-800 border-yellow-200' },
    { value: 'low', label: 'Low', color: 'bg-green-100 text-green-800 border-green-200' }
  ];
  
  // Confidentiality levels
  const confidentialityLevels = [
    { value: 'public', label: 'Public', color: 'bg-blue-100 text-blue-800' },
    { value: 'internal', label: 'Internal', color: 'bg-gray-100 text-gray-800' },
    { value: 'confidential', label: 'Confidential', color: 'bg-yellow-100 text-yellow-800' },
    { value: 'restricted', label: 'Restricted', color: 'bg-red-100 text-red-800' }
  ];
  
  // Urgency levels
  const urgencyLevels = [
    { value: 'immediate', label: 'Immediate', color: 'bg-red-100 text-red-800' },
    { value: 'high', label: 'High', color: 'bg-orange-100 text-orange-800' },
    { value: 'normal', label: 'Normal', color: 'bg-blue-100 text-blue-800' },
    { value: 'low', label: 'Low', color: 'bg-gray-100 text-gray-800' }
  ];
  
  // Watch for node changes and auto-populate form
  $: if (selectedNode) {
    autoPopulateForm(selectedNode);
  }
  
  // Track changes for auto-save
  $: if ($formData && selectedNode) {
    hasUnsavedChanges = true;
    scheduleAutoSave();
  }
  
  onMount(() => {
    return () => {
      if (autoSaveTimer) clearTimeout(autoSaveTimer);
    };
  });
  
  async function autoPopulateForm(node: unknown) {
    if (!node) return;
    
    isLoading = true;
    
    try {
      // Start with basic file information
      const newFormData = {
        title: node.name || '',
        description: '',
        tags: [],
        customTags: [],
        people: [],
        locations: [],
        organizations: [],
        dates: [],
        evidenceType: detectEvidenceType(node.type),
        legalRelevance: 'medium' as const,
        legalCategories: [],
        confidentialityLevel: 'internal' as const,
        urgencyLevel: 'normal' as const,
        keyFacts: [],
        potentialWitnesses: [],
        relatedCases: [],
        statutes: [],
        monetaryAmounts: [],
        actions: [],
        qualityScore: 0,
        extractionConfidence: {
          people: 0,
          locations: 0,
          dates: 0,
          organizations: 0
        },
        redFlags: [],
        recommendations: []
      };
      
      // If AI tags exist, populate from them
      if (node.aiTags) {
        Object.assign(newFormData, {
          title: node.aiTags.title || node.name || '',
          description: node.aiTags.summary || '',
          tags: [...(node.aiTags.tags || [])],
          people: [...(node.aiTags.people || [])],
          locations: [...(node.aiTags.locations || [])],
          organizations: [...(node.aiTags.organizations || [])],
          dates: [...(node.aiTags.dates || [])],
          evidenceType: node.aiTags.evidenceType || newFormData.evidenceType,
          legalRelevance: node.aiTags.legalRelevance || 'medium',
          legalCategories: [...(node.aiTags.legalCategories || [])],
          confidentialityLevel: node.aiTags.confidentialityLevel || 'internal',
          urgencyLevel: node.aiTags.urgencyLevel || 'normal',
          keyFacts: [...(node.aiTags.keyFacts || [])],
          potentialWitnesses: [...(node.aiTags.potentialWitnesses || [])],
          relatedCases: [...(node.aiTags.relatedCases || [])],
          statutes: [...(node.aiTags.statutes || [])],
          monetaryAmounts: [...(node.aiTags.monetaryAmounts || [])],
          actions: [...(node.aiTags.actions || [])],
          qualityScore: node.aiTags.qualityScore || 0,
          extractionConfidence: { ...newFormData.extractionConfidence, ...(node.aiTags.extractionConfidence || {}) },
          redFlags: [...(node.aiTags.redFlags || [])],
          recommendations: [...(node.aiTags.recommendations || [])]
        });
      } else {
        // Trigger AI analysis for enhanced auto-population
        await triggerEnhancedAIAnalysis(node, newFormData);
      }
      
      // Add any existing custom data
      if (node.metadata) {
        newFormData.customTags = [...(node.customTags || [])];
      }
      
      formData.set(newFormData);
      hasUnsavedChanges = false;
      
    } catch (error) {
      console.error('Auto-population failed:', error);
    } finally {
      isLoading = false;
    }
  }
  
  async function triggerEnhancedAIAnalysis(node: unknown, formData: unknown) {
    try {
      const response = await fetch('/api/ai/tag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: node.content,
          fileName: node.name,
          fileType: node.type,
          enhanced: true // Request enhanced analysis
        })
      });
      
      if (response.ok) {
        const aiTags = await response.json();
        
        // Update the node with AI tags
        node.aiTags = aiTags;
        
        // Auto-populate form with enhanced data
        Object.assign(formData, {
          title: aiTags.title || node.name,
          description: aiTags.summary || '',
          tags: [...(aiTags.tags || [])],
          people: [...(aiTags.people || [])],
          locations: [...(aiTags.locations || [])],
          organizations: [...(aiTags.organizations || [])],
          dates: [...(aiTags.dates || [])],
          evidenceType: aiTags.evidenceType || formData.evidenceType,
          legalRelevance: aiTags.legalRelevance || 'medium',
          legalCategories: [...(aiTags.legalCategories || [])],
          confidentialityLevel: aiTags.confidentialityLevel || 'internal',
          urgencyLevel: aiTags.urgencyLevel || 'normal',
          keyFacts: [...(aiTags.keyFacts || [])],
          potentialWitnesses: [...(aiTags.potentialWitnesses || [])],
          relatedCases: [...(aiTags.relatedCases || [])],
          statutes: [...(aiTags.statutes || [])],
          monetaryAmounts: [...(aiTags.monetaryAmounts || [])],
          actions: [...(aiTags.actions || [])],
          qualityScore: aiTags.qualityScore || 0,
          extractionConfidence: { ...formData.extractionConfidence, ...(aiTags.extractionConfidence || {}) },
          redFlags: [...(aiTags.redFlags || [])],
          recommendations: [...(aiTags.recommendations || [])]
        });
        
        // Notify parent components
        dispatch('aiAnalysisComplete', { node, aiTags });
      }
    } catch (error) {
      console.error('Enhanced AI analysis failed:', error);
    }
  }
  
  function detectEvidenceType(fileType: string): string {
    if (fileType.includes('image')) return 'photo';
    if (fileType.includes('video')) return 'video';
    if (fileType.includes('audio')) return 'audio';
    if (fileType.includes('pdf') || fileType.includes('document')) return 'document';
    return 'digital';
  }
  
  // Add/remove functions for arrays
  function addCustomTag() {
    if (customTag.trim() && !$formData.customTags.includes(customTag.trim())) {
      formData.update(data => ({
        ...data,
        customTags: [...data.customTags, customTag.trim()]
      }));
      customTag = '';
    }
  }
  
  function removeCustomTag(tag: string) {
    formData.update(data => ({
      ...data,
      customTags: data.customTags.filter(t => t !== tag)
    }));
  }
  
  function addCustomPerson() {
    if (customPerson.trim() && !$formData.people.includes(customPerson.trim())) {
      formData.update(data => ({
        ...data,
        people: [...data.people, customPerson.trim()]
      }));
      customPerson = '';
    }
  }
  
  function removePerson(person: string) {
    formData.update(data => ({
      ...data,
      people: data.people.filter(p => p !== person)
    }));
  }
  
  function addCustomLocation() {
    if (customLocation.trim() && !$formData.locations.includes(customLocation.trim())) {
      formData.update(data => ({
        ...data,
        locations: [...data.locations, customLocation.trim()]
      }));
      customLocation = '';
    }
  }
  
  function removeLocation(location: string) {
    formData.update(data => ({
      ...data,
      locations: data.locations.filter(l => l !== location)
    }));
  }
  
  function addCustomOrganization() {
    if (customOrganization.trim() && !$formData.organizations.includes(customOrganization.trim())) {
      formData.update(data => ({
        ...data,
        organizations: [...data.organizations, customOrganization.trim()]
      }));
      customOrganization = '';
    }
  }
  
  function removeOrganization(org: string) {
    formData.update(data => ({
      ...data,
      organizations: data.organizations.filter(o => o !== org)
    }));
  }
  
  function addCustomAction() {
    if (customAction.trim() && !$formData.actions.includes(customAction.trim())) {
      formData.update(data => ({
        ...data,
        actions: [...data.actions, customAction.trim()]
      }));
      customAction = '';
    }
  }
  
  function removeAction(action: string) {
    formData.update(data => ({
      ...data,
      actions: data.actions.filter(a => a !== action)
    }));
  }
  
  // Auto-save functionality
  function scheduleAutoSave() {
    if (autoSaveTimer) clearTimeout(autoSaveTimer);
    
    autoSaveTimer = setTimeout(() => {
      if (hasUnsavedChanges && !readOnly) {
        autoSave();
      }
    }, 5000); // Auto-save after 5 seconds of inactivity
  }
  
  async function autoSave() {
    if (!selectedNode || readOnly) return;
    
    try {
      const updatedNode = {
        ...selectedNode,
        name: $formData.title,
        title: $formData.title,
        description: $formData.description,
        customTags: $formData.customTags,
        aiTags: {
          ...selectedNode.aiTags,
          ...Object.fromEntries(
            Object.entries($formData).filter(([key]) => 
              !['customTags'].includes(key)
            )
          )
        },
        metadata: {
          ...selectedNode.metadata,
          lastModified: new Date().toISOString()
        }
      };
      
      const response = await fetch('/api/evidence/save-node', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'auto_save',
          data: updatedNode
        })
      });
      
      if (response.ok) {
        hasUnsavedChanges = false;
        lastSavedAt = new Date();
        dispatch('autoSaved', updatedNode);
      }
    } catch (error) {
      console.warn('Auto-save failed:', error);
    }
  }
  
  async function handleSave() {
    if (!selectedNode || isSaving) return;
    
    isSaving = true;
    
    try {
      const updatedNode = {
        ...selectedNode,
        name: $formData.title,
        title: $formData.title,
        description: $formData.description,
        customTags: $formData.customTags,
        aiTags: {
          ...selectedNode.aiTags,
          ...Object.fromEntries(
            Object.entries($formData).filter(([key]) => 
              !['customTags'].includes(key)
            )
          )
        },
        metadata: {
          ...selectedNode.metadata,
          lastModified: new Date().toISOString()
        }
      };
      
      const response = await fetch('/api/evidence/save-node', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'save_node',
          data: updatedNode
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        hasUnsavedChanges = false;
        lastSavedAt = new Date();
        
        dispatch('save', result.evidence);
        dispatch('showNotification', {
          type: 'success',
          message: 'Evidence saved successfully'
        });
      } else {
        throw new Error('Save failed');
      }
    } catch (error) {
      console.error('Save failed:', error);
      dispatch('showNotification', {
        type: 'error',
        message: 'Failed to save evidence'
      });
    } finally {
      isSaving = false;
    }
  }
  
  async function reanalyzeWithAI() {
    if (!selectedNode || isLoading) return;
    
    isLoading = true;
    
    try {
      // Clear existing AI tags
      selectedNode.aiTags = null;
      
      // Trigger fresh AI analysis
      await triggerEnhancedAIAnalysis(selectedNode, $formData);
      
      dispatch('showNotification', {
        type: 'success',
        message: 'AI re-analysis completed'
      });
    } catch (error) {
      console.error('Re-analysis failed:', error);
      dispatch('showNotification', {
        type: 'error',
        message: 'AI re-analysis failed'
      });
    } finally {
      isLoading = false;
    }
  }
  
  function formatDate(dateStr: string) {
    try {
      return new Date(dateStr).toLocaleDateString();
    } catch {
      return dateStr;
    }
  }
  
  function getConfidenceColor(confidence: number): string {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  }
</script>

<div class="mx-auto px-4 max-w-7xl">
  {#if selectedNode}
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <h2 class="mx-auto px-4 max-w-7xl">Evidence Inspector</h2>
        
        <!-- Action buttons -->
        <div class="mx-auto px-4 max-w-7xl">
          {#if hasUnsavedChanges}
            <span class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl"></div>
              Unsaved changes
            </span>
          {/if}
          
          {#if lastSavedAt}
            <span class="mx-auto px-4 max-w-7xl">
              Saved {new Date(lastSavedAt).toLocaleTimeString()}
            </span>
          {/if}
          
          <button 
            onclick={reanalyzeWithAI}
            disabled={isLoading}
            class="mx-auto px-4 max-w-7xl"
          >
            {#if isLoading}
              <div class="mx-auto px-4 max-w-7xl"></div>
            {:else}
              🤖
            {/if}
            Re-analyze
          </button>
        </div>
      </div>
      
      <!-- File info header -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <span class="mx-auto px-4 max-w-7xl">{evidenceTypes.find(t => t.value === $formData.evidenceType)?.icon || '📁'}</span>
          <div>
            <div class="mx-auto px-4 max-w-7xl">{selectedNode.name}</div>
            <div class="mx-auto px-4 max-w-7xl">{selectedNode.type}</div>
          </div>
        </div>
        
        {#if $formData.qualityScore > 0}
          <div class="mx-auto px-4 max-w-7xl">
            <span class="mx-auto px-4 max-w-7xl">Quality Score:</span>
            <div class="mx-auto px-4 max-w-7xl">
              <div 
                class="mx-auto px-4 max-w-7xl"
                style="width: {$formData.qualityScore * 100}%"
              ></div>
            </div>
            <span class="mx-auto px-4 max-w-7xl">{Math.round($formData.qualityScore * 100)}%</span>
          </div>
        {/if}
      </div>
      
      <!-- Loading state -->
      {#if isLoading}
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl"></div>
            <div class="mx-auto px-4 max-w-7xl">Analyzing with AI...</div>
          </div>
        </div>
      {:else}
        <!-- Form sections -->
        <div class="mx-auto px-4 max-w-7xl">
          
          <!-- Basic Information -->
          <section>
            <h3 class="mx-auto px-4 max-w-7xl">Basic Information</h3>
            <div class="mx-auto px-4 max-w-7xl">
              <div>
                <label class="mx-auto px-4 max-w-7xl">Title</label>
                <input
                  bind:value={$formData.title}
                  placeholder="Enter evidence title"
                  disabled={readOnly}
                  class="mx-auto px-4 max-w-7xl"
                />
              </div>
              
              <div>
                <label class="mx-auto px-4 max-w-7xl">Description</label>
                <textarea
                  bind:value={$formData.description}
                  placeholder="Enter description or summary"
                  disabled={readOnly}
                  rows={3}
                  class="mx-auto px-4 max-w-7xl"
                ></textarea>
              </div>
              
              <div class="mx-auto px-4 max-w-7xl">
                <div>
                  <label class="mx-auto px-4 max-w-7xl">Evidence Type</label>
                  <select 
                    bind:value={$formData.evidenceType}
                    disabled={readOnly}
                    class="mx-auto px-4 max-w-7xl"
                  >
                    {#each evidenceTypes as type}
                      <option value={type.value}>{type.icon} {type.label}</option>
                    {/each}
                  </select>
                </div>
                
                <div>
                  <label class="mx-auto px-4 max-w-7xl">Legal Relevance</label>
                  <select 
                    bind:value={$formData.legalRelevance}
                    disabled={readOnly}
                    class="mx-auto px-4 max-w-7xl"
                  >
                    {#each relevanceOptions as option}
                      <option value={option.value}>{option.label}</option>
                    {/each}
                  </select>
                </div>
              </div>
              
              <div class="mx-auto px-4 max-w-7xl">
                <div>
                  <label class="mx-auto px-4 max-w-7xl">Confidentiality</label>
                  <select 
                    bind:value={$formData.confidentialityLevel}
                    disabled={readOnly}
                    class="mx-auto px-4 max-w-7xl"
                  >
                    {#each confidentialityLevels as level}
                      <option value={level.value}>{level.label}</option>
                    {/each}
                  </select>
                </div>
                
                <div>
                  <label class="mx-auto px-4 max-w-7xl">Urgency</label>
                  <select 
                    bind:value={$formData.urgencyLevel}
                    disabled={readOnly}
                    class="mx-auto px-4 max-w-7xl"
                  >
                    {#each urgencyLevels as level}
                      <option value={level.value}>{level.label}</option>
                    {/each}
                  </select>
                </div>
              </div>
            </div>
          </section>
          
          <!-- Tags Section -->
          <section>
            <h3 class="mx-auto px-4 max-w-7xl">Tags</h3>
            <div class="mx-auto px-4 max-w-7xl">
              <!-- AI-generated tags -->
              {#if $formData.tags.length > 0}
                <div>
                  <label class="mx-auto px-4 max-w-7xl">AI-Generated Tags</label>
                  <div class="mx-auto px-4 max-w-7xl">
                    {#each $formData.tags as tag}
                      <span class="mx-auto px-4 max-w-7xl">{tag}</span>
                    {/each}
                  </div>
                </div>
              {/if}
              
              <!-- Custom tags -->
              <div>
                <label class="mx-auto px-4 max-w-7xl">Custom Tags</label>
                <div class="mx-auto px-4 max-w-7xl">
                  {#each $formData.customTags as tag}
                    <span class="mx-auto px-4 max-w-7xl">
                      {tag}
                      {#if !readOnly}
                        <button 
                          onclick={() => removeCustomTag(tag)}
                          class="mx-auto px-4 max-w-7xl"
                        >×</button>
                      {/if}
                    </span>
                  {/each}
                </div>
                
                {#if !readOnly}
                  <div class="mx-auto px-4 max-w-7xl">
                    <input
                      bind:value={customTag}
                      placeholder="Add custom tag"
                      onkeydown={(e) => e.key === 'Enter' && addCustomTag()}
                      class="mx-auto px-4 max-w-7xl"
                    />
                    <button 
                      onclick={addCustomTag} 
                      class="mx-auto px-4 max-w-7xl"
                    >Add</button>
                  </div>
                {/if}
              </div>
            </div>
          </section>
          
          <!-- Entities Section -->
          <section>
            <h3 class="mx-auto px-4 max-w-7xl">Extracted Entities</h3>
            <div class="mx-auto px-4 max-w-7xl">
              
              <!-- People -->
              {#if $formData.people.length > 0 || !readOnly}
                <div>
                  <div class="mx-auto px-4 max-w-7xl">
                    <label class="mx-auto px-4 max-w-7xl">People</label>
                    {#if $formData.extractionConfidence.people > 0}
                      <span class="mx-auto px-4 max-w-7xl">
                        {Math.round($formData.extractionConfidence.people * 100)}% confidence
                      </span>
                    {/if}
                  </div>
                  
                  <div class="mx-auto px-4 max-w-7xl">
                    {#each $formData.people as person}
                      <span class="mx-auto px-4 max-w-7xl">
                        👤 {person}
                        {#if !readOnly}
                          <button 
                            onclick={() => removePerson(person)}
                            class="mx-auto px-4 max-w-7xl"
                          >×</button>
                        {/if}
                      </span>
                    {/each}
                  </div>
                  
                  {#if !readOnly}
                    <div class="mx-auto px-4 max-w-7xl">
                      <input
                        bind:value={customPerson}
                        placeholder="Add person"
                        onkeydown={(e) => e.key === 'Enter' && addCustomPerson()}
                        class="mx-auto px-4 max-w-7xl"
                      />
                      <button 
                        onclick={addCustomPerson} 
                        class="mx-auto px-4 max-w-7xl"
                      >Add</button>
                    </div>
                  {/if}
                </div>
              {/if}
              
              <!-- Locations -->
              {#if $formData.locations.length > 0 || !readOnly}
                <div>
                  <div class="mx-auto px-4 max-w-7xl">
                    <label class="mx-auto px-4 max-w-7xl">Locations</label>
                    {#if $formData.extractionConfidence.locations > 0}
                      <span class="mx-auto px-4 max-w-7xl">
                        {Math.round($formData.extractionConfidence.locations * 100)}% confidence
                      </span>
                    {/if}
                  </div>
                  
                  <div class="mx-auto px-4 max-w-7xl">
                    {#each $formData.locations as location}
                      <span class="mx-auto px-4 max-w-7xl">
                        📍 {location}
                        {#if !readOnly}
                          <button 
                            onclick={() => removeLocation(location)}
                            class="mx-auto px-4 max-w-7xl"
                          >×</button>
                        {/if}
                      </span>
                    {/each}
                  </div>
                  
                  {#if !readOnly}
                    <div class="mx-auto px-4 max-w-7xl">
                      <input
                        bind:value={customLocation}
                        placeholder="Add location"
                        onkeydown={(e) => e.key === 'Enter' && addCustomLocation()}
                        class="mx-auto px-4 max-w-7xl"
                      />
                      <button 
                        onclick={addCustomLocation} 
                        class="mx-auto px-4 max-w-7xl"
                      >Add</button>
                    </div>
                  {/if}
                </div>
              {/if}
              
              <!-- Organizations -->
              {#if $formData.organizations.length > 0 || !readOnly}
                <div>
                  <div class="mx-auto px-4 max-w-7xl">
                    <label class="mx-auto px-4 max-w-7xl">Organizations</label>
                    {#if $formData.extractionConfidence.organizations > 0}
                      <span class="mx-auto px-4 max-w-7xl">
                        {Math.round($formData.extractionConfidence.organizations * 100)}% confidence
                      </span>
                    {/if}
                  </div>
                  
                  <div class="mx-auto px-4 max-w-7xl">
                    {#each $formData.organizations as org}
                      <span class="mx-auto px-4 max-w-7xl">
                        🏢 {org}
                        {#if !readOnly}
                          <button 
                            onclick={() => removeOrganization(org)}
                            class="mx-auto px-4 max-w-7xl"
                          >×</button>
                        {/if}
                      </span>
                    {/each}
                  </div>
                  
                  {#if !readOnly}
                    <div class="mx-auto px-4 max-w-7xl">
                      <input
                        bind:value={customOrganization}
                        placeholder="Add organization"
                        onkeydown={(e) => e.key === 'Enter' && addCustomOrganization()}
                        class="mx-auto px-4 max-w-7xl"
                      />
                      <button 
                        onclick={addCustomOrganization} 
                        class="mx-auto px-4 max-w-7xl"
                      >Add</button>
                    </div>
                  {/if}
                </div>
              {/if}
              
              <!-- Dates -->
              {#if $formData.dates.length > 0}
                <div>
                  <div class="mx-auto px-4 max-w-7xl">
                    <label class="mx-auto px-4 max-w-7xl">Dates</label>
                    {#if $formData.extractionConfidence.dates > 0}
                      <span class="mx-auto px-4 max-w-7xl">
                        {Math.round($formData.extractionConfidence.dates * 100)}% confidence
                      </span>
                    {/if}
                  </div>
                  
                  <div class="mx-auto px-4 max-w-7xl">
                    {#each $formData.dates as date}
                      <span class="mx-auto px-4 max-w-7xl">
                        📅 {formatDate(date)}
                      </span>
                    {/each}
                  </div>
                </div>
              {/if}
            </div>
          </section>
          
          <!-- Key Facts -->
          {#if $formData.keyFacts.length > 0}
            <section>
              <h3 class="mx-auto px-4 max-w-7xl">Key Facts</h3>
              <ul class="mx-auto px-4 max-w-7xl">
                {#each $formData.keyFacts as fact}
                  <li class="mx-auto px-4 max-w-7xl">
                    <span class="mx-auto px-4 max-w-7xl">•</span>
                    <span class="mx-auto px-4 max-w-7xl">{fact}</span>
                  </li>
                {/each}
              </ul>
            </section>
          {/if}
          
          <!-- Actions & Recommendations -->
          {#if $formData.actions.length > 0 || $formData.recommendations.length > 0 || !readOnly}
            <section>
              <h3 class="mx-auto px-4 max-w-7xl">Actions & Recommendations</h3>
              <div class="mx-auto px-4 max-w-7xl">
                
                <!-- Action Items -->
                <div>
                  <label class="mx-auto px-4 max-w-7xl">Action Items</label>
                  <div class="mx-auto px-4 max-w-7xl">
                    {#each $formData.actions as action}
                      <span class="mx-auto px-4 max-w-7xl">
                        ⚡ {action}
                        {#if !readOnly}
                          <button 
                            onclick={() => removeAction(action)}
                            class="mx-auto px-4 max-w-7xl"
                          >×</button>
                        {/if}
                      </span>
                    {/each}
                  </div>
                  
                  {#if !readOnly}
                    <div class="mx-auto px-4 max-w-7xl">
                      <input
                        bind:value={customAction}
                        placeholder="Add action item"
                        onkeydown={(e) => e.key === 'Enter' && addCustomAction()}
                        class="mx-auto px-4 max-w-7xl"
                      />
                      <button 
                        onclick={addCustomAction} 
                        class="mx-auto px-4 max-w-7xl"
                      >Add</button>
                    </div>
                  {/if}
                </div>
                
                <!-- AI Recommendations -->
                {#if $formData.recommendations.length > 0}
                  <div>
                    <label class="mx-auto px-4 max-w-7xl">AI Recommendations</label>
                    <ul class="mx-auto px-4 max-w-7xl">
                      {#each $formData.recommendations as recommendation}
                        <li class="mx-auto px-4 max-w-7xl">
                          <span class="mx-auto px-4 max-w-7xl">💡</span>
                          <span>{recommendation}</span>
                        </li>
                      {/each}
                    </ul>
                  </div>
                {/if}
              </div>
            </section>
          {/if}
          
          <!-- Red Flags -->
          {#if $formData.redFlags.length > 0}
            <section>
              <h3 class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">⚠️</span>
                Red Flags
              </h3>
              <div class="mx-auto px-4 max-w-7xl">
                {#each $formData.redFlags as flag}
                  <div class="mx-auto px-4 max-w-7xl">
                    <span class="mx-auto px-4 max-w-7xl">{flag}</span>
                  </div>
                {/each}
              </div>
            </section>
          {/if}
          
          <!-- Additional Legal Information -->
          {#if $formData.statutes.length > 0 || $formData.monetaryAmounts.length > 0 || $formData.potentialWitnesses.length > 0 || $formData.relatedCases.length > 0}
            <section>
              <h3 class="mx-auto px-4 max-w-7xl">Legal Information</h3>
              <div class="mx-auto px-4 max-w-7xl">
                
                {#if $formData.statutes.length > 0}
                  <div>
                    <label class="mx-auto px-4 max-w-7xl">Relevant Statutes</label>
                    <div class="mx-auto px-4 max-w-7xl">
                      {#each $formData.statutes as statute}
                        <span class="mx-auto px-4 max-w-7xl">⚖️ {statute}</span>
                      {/each}
                    </div>
                  </div>
                {/if}
                
                {#if $formData.monetaryAmounts.length > 0}
                  <div>
                    <label class="mx-auto px-4 max-w-7xl">Monetary Amounts</label>
                    <div class="mx-auto px-4 max-w-7xl">
                      {#each $formData.monetaryAmounts as amount}
                        <span class="mx-auto px-4 max-w-7xl">💰 {amount}</span>
                      {/each}
                    </div>
                  </div>
                {/if}
                
                {#if $formData.potentialWitnesses.length > 0}
                  <div>
                    <label class="mx-auto px-4 max-w-7xl">Potential Witnesses</label>
                    <div class="mx-auto px-4 max-w-7xl">
                      {#each $formData.potentialWitnesses as witness}
                        <span class="mx-auto px-4 max-w-7xl">👁️ {witness}</span>
                      {/each}
                    </div>
                  </div>
                {/if}
                
                {#if $formData.relatedCases.length > 0}
                  <div>
                    <label class="mx-auto px-4 max-w-7xl">Related Cases</label>
                    <div class="mx-auto px-4 max-w-7xl">
                      {#each $formData.relatedCases as case_ref}
                        <span class="mx-auto px-4 max-w-7xl">📁 {case_ref}</span>
                      {/each}
                    </div>
                  </div>
                {/if}
              </div>
            </section>
          {/if}
        </div>
        
        <!-- Save button -->
        {#if !readOnly}
          <div class="mx-auto px-4 max-w-7xl">
            <button 
              onclick={handleSave}
              disabled={isSaving || !hasUnsavedChanges}
              class="mx-auto px-4 max-w-7xl"
            >
              {#if isSaving}
                <div class="mx-auto px-4 max-w-7xl"></div>
                Saving...
              {:else}
                💾 Save Evidence
              {/if}
            </button>
          </div>
        {/if}
      {/if}
    </div>
  {:else}
    <!-- No selection state -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">📋</div>
      <div class="mx-auto px-4 max-w-7xl">Enhanced Inspector</div>
      <div class="mx-auto px-4 max-w-7xl">Select evidence to view AI-powered analysis and auto-populated fields</div>
      <div class="mx-auto px-4 max-w-7xl">Powered by advanced natural language processing</div>
    </div>
  {/if}
</div>

<style>
  .enhanced-inspector-panel {
    min-height: 100%;
    max-height: 100vh;
    overflow-y: auto;
  }
</style>

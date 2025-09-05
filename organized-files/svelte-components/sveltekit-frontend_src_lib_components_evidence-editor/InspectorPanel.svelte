<script lang="ts">
  interface Props {
    onaiAnalysisComplete?: (event?: any) => void;
    onautoSaved?: (event?: any) => void;
    onsave?: (event?: any) => void;
    onshowNotification?: (event?: any) => void;
  }
  let {
    selectedNode = null,
    readOnly = false
  } = $props();



    import { writable } from 'svelte/store';
  
    
      
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
  $effect(() => { if (selectedNode) {
    autoPopulateForm(selectedNode);
}
  // Track changes for auto-save
  $effect(() => { if ($formData && selectedNode) {
    hasUnsavedChanges = true;
    scheduleAutoSave();
}
  onMount(() => {
    return () => {
      if (autoSaveTimer) clearTimeout(autoSaveTimer);
    };
  });
  
  async function autoPopulateForm(node: any) {
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
  async function triggerEnhancedAIAnalysis(node: any, formData: any) {
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
        onaiAnalysisComplete?.();
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
        onautoSaved?.();
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
        
        onsave?.();
        onshowNotification?.();
      } else {
        throw new Error('Save failed');
}
    } catch (error) {
      console.error('Save failed:', error);
      onshowNotification?.();
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
      
      onshowNotification?.();
    } catch (error) {
      console.error('Re-analysis failed:', error);
      onshowNotification?.();
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

<div class="space-y-4">
  {#if selectedNode}
    <div class="space-y-4">
      <div class="space-y-4">
        <h2 class="space-y-4">Evidence Inspector</h2>
        
        <!-- Action buttons -->
        <div class="space-y-4">
          {#if hasUnsavedChanges}
            <span class="space-y-4">
              <div class="space-y-4"></div>
              Unsaved changes
            </span>
          {/if}
          
          {#if lastSavedAt}
            <span class="space-y-4">
              Saved {new Date(lastSavedAt).toLocaleTimeString()}
            </span>
          {/if}
          
          <button 
            onclick={reanalyzeWithAI}
            disabled={isLoading}
            class="space-y-4"
          >
            {#if isLoading}
              <div class="space-y-4"></div>
            {:else}
              🤖
            {/if}
            Re-analyze
          </button>
        </div>
      </div>
      
      <!-- File info header -->
      <div class="space-y-4">
        <div class="space-y-4">
          <span class="space-y-4">{evidenceTypes.find(t => t.value === $formData.evidenceType)?.icon || '📁'}</span>
          <div>
            <div class="space-y-4">{selectedNode.name}</div>
            <div class="space-y-4">{selectedNode.type}</div>
          </div>
        </div>
        
        {#if $formData.qualityScore > 0}
          <div class="space-y-4">
            <span class="space-y-4">Quality Score:</span>
            <div class="space-y-4">
              <div 
                class="space-y-4"
                style="width: {$formData.qualityScore * 100}%"
              ></div>
            </div>
            <span class="space-y-4">{Math.round($formData.qualityScore * 100)}%</span>
          </div>
        {/if}
      </div>
      
      <!-- Loading state -->
      {#if isLoading}
        <div class="space-y-4">
          <div class="space-y-4">
            <div class="space-y-4"></div>
            <div class="space-y-4">Analyzing with AI...</div>
          </div>
        </div>
      {:else}
        <!-- Form sections -->
        <div class="space-y-4">
          
          <!-- Basic Information -->
          <section>
            <h3 class="space-y-4">Basic Information</h3>
            <div class="space-y-4">
              <div>
                <label class="space-y-4">Title</label>
                <input
                  bind:value={$formData.title}
                  placeholder="Enter evidence title"
                  disabled={readOnly}
                  class="space-y-4"
                />
              </div>
              
              <div>
                <label class="space-y-4">Description</label>
                <textarea
                  bind:value={$formData.description}
                  placeholder="Enter description or summary"
                  disabled={readOnly}
                  rows={3}
                  class="space-y-4"
                ></textarea>
              </div>
              
              <div class="space-y-4">
                <div>
                  <label class="space-y-4">Evidence Type</label>
                  <select 
                    bind:value={$formData.evidenceType}
                    disabled={readOnly}
                    class="space-y-4"
                  >
                    {#each evidenceTypes as type}
                      <option value={type.value}>{type.icon} {type.label}</option>
                    {/each}
                  </select>
                </div>
                
                <div>
                  <label class="space-y-4">Legal Relevance</label>
                  <select 
                    bind:value={$formData.legalRelevance}
                    disabled={readOnly}
                    class="space-y-4"
                  >
                    {#each relevanceOptions as option}
                      <option value={option.value}>{option.label}</option>
                    {/each}
                  </select>
                </div>
              </div>
              
              <div class="space-y-4">
                <div>
                  <label class="space-y-4">Confidentiality</label>
                  <select 
                    bind:value={$formData.confidentialityLevel}
                    disabled={readOnly}
                    class="space-y-4"
                  >
                    {#each confidentialityLevels as level}
                      <option value={level.value}>{level.label}</option>
                    {/each}
                  </select>
                </div>
                
                <div>
                  <label class="space-y-4">Urgency</label>
                  <select 
                    bind:value={$formData.urgencyLevel}
                    disabled={readOnly}
                    class="space-y-4"
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
            <h3 class="space-y-4">Tags</h3>
            <div class="space-y-4">
              <!-- AI-generated tags -->
              {#if $formData.tags.length > 0}
                <div>
                  <label class="space-y-4">AI-Generated Tags</label>
                  <div class="space-y-4">
                    {#each $formData.tags as tag}
                      <span class="space-y-4">{tag}</span>
                    {/each}
                  </div>
                </div>
              {/if}
              
              <!-- Custom tags -->
              <div>
                <label class="space-y-4">Custom Tags</label>
                <div class="space-y-4">
                  {#each $formData.customTags as tag}
                    <span class="space-y-4">
                      {tag}
                      {#if !readOnly}
                        <button 
                          onclick={() => removeCustomTag(tag)}
                          class="space-y-4"
                        >×</button>
                      {/if}
                    </span>
                  {/each}
                </div>
                
                {#if !readOnly}
                  <div class="space-y-4">
                    <input
                      bind:value={customTag}
                      placeholder="Add custom tag"
                      onkeydown={(e) => e.key === 'Enter' && addCustomTag()}
                      class="space-y-4"
                    />
                    <button 
                      onclick={addCustomTag} 
                      class="space-y-4"
                    >Add</button>
                  </div>
                {/if}
              </div>
            </div>
          </section>
          
          <!-- Entities Section -->
          <section>
            <h3 class="space-y-4">Extracted Entities</h3>
            <div class="space-y-4">
              
              <!-- People -->
              {#if $formData.people.length > 0 || !readOnly}
                <div>
                  <div class="space-y-4">
                    <label class="space-y-4">People</label>
                    {#if $formData.extractionConfidence.people > 0}
                      <span class="space-y-4">
                        {Math.round($formData.extractionConfidence.people * 100)}% confidence
                      </span>
                    {/if}
                  </div>
                  
                  <div class="space-y-4">
                    {#each $formData.people as person}
                      <span class="space-y-4">
                        👤 {person}
                        {#if !readOnly}
                          <button 
                            onclick={() => removePerson(person)}
                            class="space-y-4"
                          >×</button>
                        {/if}
                      </span>
                    {/each}
                  </div>
                  
                  {#if !readOnly}
                    <div class="space-y-4">
                      <input
                        bind:value={customPerson}
                        placeholder="Add person"
                        onkeydown={(e) => e.key === 'Enter' && addCustomPerson()}
                        class="space-y-4"
                      />
                      <button 
                        onclick={addCustomPerson} 
                        class="space-y-4"
                      >Add</button>
                    </div>
                  {/if}
                </div>
              {/if}
              
              <!-- Locations -->
              {#if $formData.locations.length > 0 || !readOnly}
                <div>
                  <div class="space-y-4">
                    <label class="space-y-4">Locations</label>
                    {#if $formData.extractionConfidence.locations > 0}
                      <span class="space-y-4">
                        {Math.round($formData.extractionConfidence.locations * 100)}% confidence
                      </span>
                    {/if}
                  </div>
                  
                  <div class="space-y-4">
                    {#each $formData.locations as location}
                      <span class="space-y-4">
                        📍 {location}
                        {#if !readOnly}
                          <button 
                            onclick={() => removeLocation(location)}
                            class="space-y-4"
                          >×</button>
                        {/if}
                      </span>
                    {/each}
                  </div>
                  
                  {#if !readOnly}
                    <div class="space-y-4">
                      <input
                        bind:value={customLocation}
                        placeholder="Add location"
                        onkeydown={(e) => e.key === 'Enter' && addCustomLocation()}
                        class="space-y-4"
                      />
                      <button 
                        onclick={addCustomLocation} 
                        class="space-y-4"
                      >Add</button>
                    </div>
                  {/if}
                </div>
              {/if}
              
              <!-- Organizations -->
              {#if $formData.organizations.length > 0 || !readOnly}
                <div>
                  <div class="space-y-4">
                    <label class="space-y-4">Organizations</label>
                    {#if $formData.extractionConfidence.organizations > 0}
                      <span class="space-y-4">
                        {Math.round($formData.extractionConfidence.organizations * 100)}% confidence
                      </span>
                    {/if}
                  </div>
                  
                  <div class="space-y-4">
                    {#each $formData.organizations as org}
                      <span class="space-y-4">
                        🏢 {org}
                        {#if !readOnly}
                          <button 
                            onclick={() => removeOrganization(org)}
                            class="space-y-4"
                          >×</button>
                        {/if}
                      </span>
                    {/each}
                  </div>
                  
                  {#if !readOnly}
                    <div class="space-y-4">
                      <input
                        bind:value={customOrganization}
                        placeholder="Add organization"
                        onkeydown={(e) => e.key === 'Enter' && addCustomOrganization()}
                        class="space-y-4"
                      />
                      <button 
                        onclick={addCustomOrganization} 
                        class="space-y-4"
                      >Add</button>
                    </div>
                  {/if}
                </div>
              {/if}
              
              <!-- Dates -->
              {#if $formData.dates.length > 0}
                <div>
                  <div class="space-y-4">
                    <label class="space-y-4">Dates</label>
                    {#if $formData.extractionConfidence.dates > 0}
                      <span class="space-y-4">
                        {Math.round($formData.extractionConfidence.dates * 100)}% confidence
                      </span>
                    {/if}
                  </div>
                  
                  <div class="space-y-4">
                    {#each $formData.dates as date}
                      <span class="space-y-4">
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
              <h3 class="space-y-4">Key Facts</h3>
              <ul class="space-y-4">
                {#each $formData.keyFacts as fact}
                  <li class="space-y-4">
                    <span class="space-y-4">•</span>
                    <span class="space-y-4">{fact}</span>
                  </li>
                {/each}
              </ul>
            </section>
          {/if}
          
          <!-- Actions & Recommendations -->
          {#if $formData.actions.length > 0 || $formData.recommendations.length > 0 || !readOnly}
            <section>
              <h3 class="space-y-4">Actions & Recommendations</h3>
              <div class="space-y-4">
                
                <!-- Action Items -->
                <div>
                  <label class="space-y-4">Action Items</label>
                  <div class="space-y-4">
                    {#each $formData.actions as action}
                      <span class="space-y-4">
                        ⚡ {action}
                        {#if !readOnly}
                          <button 
                            onclick={() => removeAction(action)}
                            class="space-y-4"
                          >×</button>
                        {/if}
                      </span>
                    {/each}
                  </div>
                  
                  {#if !readOnly}
                    <div class="space-y-4">
                      <input
                        bind:value={customAction}
                        placeholder="Add action item"
                        onkeydown={(e) => e.key === 'Enter' && addCustomAction()}
                        class="space-y-4"
                      />
                      <button 
                        onclick={addCustomAction} 
                        class="space-y-4"
                      >Add</button>
                    </div>
                  {/if}
                </div>
                
                <!-- AI Recommendations -->
                {#if $formData.recommendations.length > 0}
                  <div>
                    <label class="space-y-4">AI Recommendations</label>
                    <ul class="space-y-4">
                      {#each $formData.recommendations as recommendation}
                        <li class="space-y-4">
                          <span class="space-y-4">💡</span>
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
              <h3 class="space-y-4">
                <span class="space-y-4">⚠️</span>
                Red Flags
              </h3>
              <div class="space-y-4">
                {#each $formData.redFlags as flag}
                  <div class="space-y-4">
                    <span class="space-y-4">{flag}</span>
                  </div>
                {/each}
              </div>
            </section>
          {/if}
          
          <!-- Additional Legal Information -->
          {#if $formData.statutes.length > 0 || $formData.monetaryAmounts.length > 0 || $formData.potentialWitnesses.length > 0 || $formData.relatedCases.length > 0}
            <section>
              <h3 class="space-y-4">Legal Information</h3>
              <div class="space-y-4">
                
                {#if $formData.statutes.length > 0}
                  <div>
                    <label class="space-y-4">Relevant Statutes</label>
                    <div class="space-y-4">
                      {#each $formData.statutes as statute}
                        <span class="space-y-4">⚖️ {statute}</span>
                      {/each}
                    </div>
                  </div>
                {/if}
                
                {#if $formData.monetaryAmounts.length > 0}
                  <div>
                    <label class="space-y-4">Monetary Amounts</label>
                    <div class="space-y-4">
                      {#each $formData.monetaryAmounts as amount}
                        <span class="space-y-4">💰 {amount}</span>
                      {/each}
                    </div>
                  </div>
                {/if}
                
                {#if $formData.potentialWitnesses.length > 0}
                  <div>
                    <label class="space-y-4">Potential Witnesses</label>
                    <div class="space-y-4">
                      {#each $formData.potentialWitnesses as witness}
                        <span class="space-y-4">👁️ {witness}</span>
                      {/each}
                    </div>
                  </div>
                {/if}
                
                {#if $formData.relatedCases.length > 0}
                  <div>
                    <label class="space-y-4">Related Cases</label>
                    <div class="space-y-4">
                      {#each $formData.relatedCases as case_ref}
                        <span class="space-y-4">📁 {case_ref}</span>
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
          <div class="space-y-4">
            <button 
              onclick={handleSave}
              disabled={isSaving || !hasUnsavedChanges}
              class="space-y-4"
            >
              {#if isSaving}
                <div class="space-y-4"></div>
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
    <div class="space-y-4">
      <div class="space-y-4">📋</div>
      <div class="space-y-4">Enhanced Inspector</div>
      <div class="space-y-4">Select evidence to view AI-powered analysis and auto-populated fields</div>
      <div class="space-y-4">Powered by advanced natural language processing</div>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  .enhanced-inspector-panel {
    min-height: 100%;
    max-height: 100vh;
    overflow-y: auto
}
</style>

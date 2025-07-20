<!-- Evidence Analysis Modal with LLM integration -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import Dialog from '$lib/components/ui/dialog/Dialog.svelte';
  import Grid from '$lib/components/ui/grid/Grid.svelte';
  import GridItem from '$lib/components/ui/grid/GridItem.svelte';
  import Button from "$lib/components/ui/button";
  import Input from '$lib/components/ui/Input.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  // Icons
  import { FileText, Brain, Tag, Scale, Zap, Download, Upload, Sparkles } from 'lucide-svelte';

  export let open: boolean = false;
  export let evidence: {
    id: string;
    content: string;
    type: string;
    caseId?: string;
    metadata?: any;
    analysis?: {
      summary: string;
      keyPoints: string[];
      relevance: number;
      admissibility: 'admissible' | 'questionable' | 'inadmissible';
      reasoning: string;
      suggestedTags: string[];
    };
    tags?: string[];
    similarEvidence?: Array<{
      id: string;
      content: string;
      similarity: number;
    }>;
  } | null = null;

  const dispatch = createEventDispatcher();

  let isAnalyzing = false;
  let newTags: string = '';
  let analysisMode: 'quick' | 'detailed' | 'legal' = 'detailed';

  async function analyzeEvidence() {
    if (!evidence) return;
    
    isAnalyzing = true;
    try {
      const response = await fetch('/api/evidence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          caseId: evidence.caseId,
          content: evidence.content,
          type: evidence.evidenceType || evidence.type,
          generateAnalysis: true,
          metadata: { analysisMode }
        })
      });

      const result = await response.json();
      if (result.success) {
        evidence = { ...evidence, ...result.evidence };
        dispatch('evidenceUpdated', evidence);}
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      isAnalyzing = false;}}
  async function updateTags() {
    if (!evidence || !newTags.trim()) return;
    
    const tags = newTags.split(',').map(t => t.trim()).filter(Boolean);
    
    try {
      const response = await fetch('/api/evidence', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evidenceId: evidence.id,
          caseId: evidence.caseId,
          tags: [...(evidence.tags || []), ...tags]
        })
      });

      const result = await response.json();
      if (result.success) {
        evidence = { ...evidence, tags: result.evidence.tags };
        newTags = '';
        dispatch('evidenceUpdated', evidence);}
    } catch (error) {
      console.error('Tag update failed:', error);}}
  function getAdmissibilityColor(admissibility: string): string {
    switch (admissibility) {
      case 'admissible': return 'bg-green-100 text-green-800 border-green-300';
      case 'questionable': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'inadmissible': return 'bg-red-100 text-red-800 border-red-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';}}
  function getRelevanceColor(relevance: number): string {
    if (relevance >= 8) return 'text-green-600';
    if (relevance >= 6) return 'text-yellow-600';
    return 'text-red-600';}
</script>

<Dialog 
  bind:open 
  title="Evidence Analysis" 
  description="AI-powered legal evidence analysis and tagging"
  size="xl"
>
  <svelte:fragment slot="trigger">
    <slot name="trigger" />
  </svelte:fragment>

  {#if evidence}
    <div class="container mx-auto px-4">
      <!-- Evidence Header -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4">
            <FileText class="container mx-auto px-4" />
          </div>
          <div>
            <h3 class="container mx-auto px-4">
              {evidence.evidenceType || evidence.type} Evidence
            </h3>
            <p class="container mx-auto px-4">ID: {evidence.id}</p>
          </div>
        </div>
        
        <div class="container mx-auto px-4">
          <Button variant="secondary" size="sm">
            <Download class="container mx-auto px-4" />
            Export
          </Button>
          <Button 
            variant="primary" 
            size="sm" 
            on:click={() => analyzeEvidence()}
            disabled={isAnalyzing}
          >
            {#if isAnalyzing}
              <div class="container mx-auto px-4"></div>
              Analyzing...
            {:else}
              <Brain class="container mx-auto px-4" />
              Re-analyze
            {/if}
          </Button>
        </div>
      </div>

      <!-- Grid Layout -->
      <Grid columns={12} gap="md" responsive={true}>
        
        <!-- Evidence Content -->
        <GridItem colSpan={8}>
          <div class="container mx-auto px-4">
            <h4 class="container mx-auto px-4">Evidence Content</h4>
            <p class="container mx-auto px-4">
              {evidence.content}
            </p>
          </div>
        </GridItem>

        <!-- Quick Stats -->
        <GridItem colSpan={4}>
          <div class="container mx-auto px-4">
            <!-- Relevance Score -->
            {#if evidence.analysis?.relevance}
              <div class="container mx-auto px-4">
                <div class="container mx-auto px-4">
                  <span class="container mx-auto px-4">Relevance Score</span>
                  <Scale class="container mx-auto px-4" />
                </div>
                <div class="container mx-auto px-4">
                  {evidence.analysis.relevance}/10
                </div>
              </div>
            {/if}

            <!-- Admissibility -->
            {#if evidence.analysis?.admissibility}
              <div class="container mx-auto px-4">
                <div class="container mx-auto px-4">
                  <span class="container mx-auto px-4">Admissibility</span>
                  <Zap class="container mx-auto px-4" />
                </div>
                <span class="container mx-auto px-4">
                  {evidence.analysis.admissibility}
                </span>
              </div>
            {/if}
          </div>
        </GridItem>

        <!-- Analysis Section -->
        {#if evidence.analysis}
          <GridItem colSpan={12}>
            <div class="container mx-auto px-4">
              <h4 class="container mx-auto px-4">
                <Sparkles class="container mx-auto px-4" />
                AI Analysis
              </h4>
              
              <Grid columns={12} gap="md">
                <!-- Summary -->
                <GridItem colSpan={6}>
                  <div>
                    <h5 class="container mx-auto px-4">Summary</h5>
                    <p class="container mx-auto px-4">
                      {evidence.analysis.summary}
                    </p>
                  </div>
                </GridItem>

                <!-- Key Points -->
                <GridItem colSpan={6}>
                  <div>
                    <h5 class="container mx-auto px-4">Key Points</h5>
                    <ul class="container mx-auto px-4">
                      {#each evidence.analysis.keyPoints as point}
                        <li class="container mx-auto px-4">
                          <span class="container mx-auto px-4"></span>
                          {point}
                        </li>
                      {/each}
                    </ul>
                  </div>
                </GridItem>

                <!-- Legal Reasoning -->
                <GridItem colSpan={12}>
                  <div>
                    <h5 class="container mx-auto px-4">Legal Reasoning</h5>
                    <p class="container mx-auto px-4">
                      {evidence.analysis.reasoning}
                    </p>
                  </div>
                </GridItem>
              </Grid>
            </div>
          </GridItem>
        {/if}

        <!-- Tags Section -->
        <GridItem colSpan={8}>
          <div class="container mx-auto px-4">
            <h4 class="container mx-auto px-4">
              <Tag class="container mx-auto px-4" />
              Tags
            </h4>
            
            <!-- Existing Tags -->
            <div class="container mx-auto px-4">
              {#each evidence.tags || [] as tag}
                <Badge variant="secondary">{tag}</Badge>
              {/each}
              {#each evidence.analysis?.suggestedTags || [] as tag}
                <Badge variant="secondary" class="container mx-auto px-4">
                  {tag} <span class="container mx-auto px-4">(suggested)</span>
                </Badge>
              {/each}
            </div>

            <!-- Add Tags -->
            <div class="container mx-auto px-4">
              <Input
                bind:value={newTags}
                placeholder="Add tags (comma-separated)"
                class="container mx-auto px-4"
              />
              <Button size="sm" on:click={() => updateTags()} disabled={!newTags.trim()}>
                Add
              </Button>
            </div>
          </div>
        </GridItem>

        <!-- Similar Evidence -->
        <GridItem colSpan={4}>
          <div class="container mx-auto px-4">
            <h4 class="container mx-auto px-4">Similar Evidence</h4>
            <div class="container mx-auto px-4">
              {#each evidence.similarEvidence || [] as similar}
                <div class="container mx-auto px-4">
                  <div class="container mx-auto px-4">Similarity: {(similar.similarity * 100).toFixed(0)}%</div>
                  <p class="container mx-auto px-4">
                    {similar.content.substring(0, 80)}...
                  </p>
                </div>
              {/each}
            </div>
          </div>
        </GridItem>
      </Grid>
    </div>
  {/if}

  <svelte:fragment slot="footer" let:close>
    <Button variant="secondary" on:click={() => close()}>
      Close
    </Button>
    <Button variant="primary" on:click={() => dispatch('saveAnalysis', evidence)}>
      Save Analysis
    </Button>
  </svelte:fragment>
</Dialog>

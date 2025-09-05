<!-- Evidence Analysis Modal with LLM integration -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import Dialog from '../../../lib/components/ui/dialog/Dialog.svelte';
  import Grid from '../../../lib/components/ui/grid/Grid.svelte';
  import GridItem from '../../../lib/components/ui/grid/GridItem.svelte';
  import { Button } from "$lib/components/ui/button";
  import Input from '../../../lib/components/ui/Input.svelte';
  import Badge from '../../../lib/components/ui/Badge.svelte';
  
  // Icons
  import { FileText, Brain, Tag, Scale, Zap, Download, Upload, Sparkles } from 'lucide-svelte';

  export let open: boolean = false;
  export let evidence: {
    id: string;
    content: string;
    type: string;
    caseId?: string;
    metadata?: unknown;
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
        dispatch('evidenceUpdated', evidence);
      }
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      isAnalyzing = false;
    }
  }

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
        dispatch('evidenceUpdated', evidence);
      }
    } catch (error) {
      console.error('Tag update failed:', error);
    }
  }

  function getAdmissibilityColor(admissibility: string): string {
    switch (admissibility) {
      case 'admissible': return 'bg-green-100 text-green-800 border-green-300';
      case 'questionable': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'inadmissible': return 'bg-red-100 text-red-800 border-red-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  }

  function getRelevanceColor(relevance: number): string {
    if (relevance >= 8) return 'text-green-600';
    if (relevance >= 6) return 'text-yellow-600';
    return 'text-red-600';
  }
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
    <div class="mx-auto px-4 max-w-7xl">
      <!-- Evidence Header -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <FileText class="mx-auto px-4 max-w-7xl" />
          </div>
          <div>
            <h3 class="mx-auto px-4 max-w-7xl">
              {evidence.evidenceType || evidence.type} Evidence
            </h3>
            <p class="mx-auto px-4 max-w-7xl">ID: {evidence.id}</p>
          </div>
        </div>
        
        <div class="mx-auto px-4 max-w-7xl">
          <Button variant="secondary" size="sm">
            <Download class="mx-auto px-4 max-w-7xl" />
            Export
          </Button>
          <Button 
            variant="primary" 
            size="sm" 
            onclick={() => analyzeEvidence()}
            disabled={isAnalyzing}
          >
            {#if isAnalyzing}
              <div class="mx-auto px-4 max-w-7xl"></div>
              Analyzing...
            {:else}
              <Brain class="mx-auto px-4 max-w-7xl" />
              Re-analyze
            {/if}
          </Button>
        </div>
      </div>

      <!-- Grid Layout -->
      <Grid columns={12} gap="md" responsive={true}>
        
        <!-- Evidence Content -->
        <GridItem colSpan={8}>
          <div class="mx-auto px-4 max-w-7xl">
            <h4 class="mx-auto px-4 max-w-7xl">Evidence Content</h4>
            <p class="mx-auto px-4 max-w-7xl">
              {evidence.content}
            </p>
          </div>
        </GridItem>

        <!-- Quick Stats -->
        <GridItem colSpan={4}>
          <div class="mx-auto px-4 max-w-7xl">
            <!-- Relevance Score -->
            {#if evidence.analysis?.relevance}
              <div class="mx-auto px-4 max-w-7xl">
                <div class="mx-auto px-4 max-w-7xl">
                  <span class="mx-auto px-4 max-w-7xl">Relevance Score</span>
                  <Scale class="mx-auto px-4 max-w-7xl" />
                </div>
                <div class="mx-auto px-4 max-w-7xl">
                  {evidence.analysis.relevance}/10
                </div>
              </div>
            {/if}

            <!-- Admissibility -->
            {#if evidence.analysis?.admissibility}
              <div class="mx-auto px-4 max-w-7xl">
                <div class="mx-auto px-4 max-w-7xl">
                  <span class="mx-auto px-4 max-w-7xl">Admissibility</span>
                  <Zap class="mx-auto px-4 max-w-7xl" />
                </div>
                <span class="mx-auto px-4 max-w-7xl">
                  {evidence.analysis.admissibility}
                </span>
              </div>
            {/if}
          </div>
        </GridItem>

        <!-- Analysis Section -->
        {#if evidence.analysis}
          <GridItem colSpan={12}>
            <div class="mx-auto px-4 max-w-7xl">
              <h4 class="mx-auto px-4 max-w-7xl">
                <Sparkles class="mx-auto px-4 max-w-7xl" />
                AI Analysis
              </h4>
              
              <Grid columns={12} gap="md">
                <!-- Summary -->
                <GridItem colSpan={6}>
                  <div>
                    <h5 class="mx-auto px-4 max-w-7xl">Summary</h5>
                    <p class="mx-auto px-4 max-w-7xl">
                      {evidence.analysis.summary}
                    </p>
                  </div>
                </GridItem>

                <!-- Key Points -->
                <GridItem colSpan={6}>
                  <div>
                    <h5 class="mx-auto px-4 max-w-7xl">Key Points</h5>
                    <ul class="mx-auto px-4 max-w-7xl">
                      {#each evidence.analysis.keyPoints as point}
                        <li class="mx-auto px-4 max-w-7xl">
                          <span class="mx-auto px-4 max-w-7xl"></span>
                          {point}
                        </li>
                      {/each}
                    </ul>
                  </div>
                </GridItem>

                <!-- Legal Reasoning -->
                <GridItem colSpan={12}>
                  <div>
                    <h5 class="mx-auto px-4 max-w-7xl">Legal Reasoning</h5>
                    <p class="mx-auto px-4 max-w-7xl">
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
          <div class="mx-auto px-4 max-w-7xl">
            <h4 class="mx-auto px-4 max-w-7xl">
              <Tag class="mx-auto px-4 max-w-7xl" />
              Tags
            </h4>
            
            <!-- Existing Tags -->
            <div class="mx-auto px-4 max-w-7xl">
              {#each evidence.tags || [] as tag}
                <Badge variant="secondary">{tag}</Badge>
              {/each}
              {#each evidence.analysis?.suggestedTags || [] as tag}
                <Badge variant="secondary" class="mx-auto px-4 max-w-7xl">
                  {tag} <span class="mx-auto px-4 max-w-7xl">(suggested)</span>
                </Badge>
              {/each}
            </div>

            <!-- Add Tags -->
            <div class="mx-auto px-4 max-w-7xl">
              <Input
                bind:value={newTags}
                placeholder="Add tags (comma-separated)"
                class="mx-auto px-4 max-w-7xl"
              />
              <Button size="sm" onclick={() => updateTags()} disabled={!newTags.trim()}>
                Add
              </Button>
            </div>
          </div>
        </GridItem>

        <!-- Similar Evidence -->
        <GridItem colSpan={4}>
          <div class="mx-auto px-4 max-w-7xl">
            <h4 class="mx-auto px-4 max-w-7xl">Similar Evidence</h4>
            <div class="mx-auto px-4 max-w-7xl">
              {#each evidence.similarEvidence || [] as similar}
                <div class="mx-auto px-4 max-w-7xl">
                  <div class="mx-auto px-4 max-w-7xl">Similarity: {(similar.similarity * 100).toFixed(0)}%</div>
                  <p class="mx-auto px-4 max-w-7xl">
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
    <Button variant="secondary" onclick={() => close()}>
      Close
    </Button>
    <Button variant="primary" onclick={() => dispatch('saveAnalysis', evidence)}>
      Save Analysis
    </Button>
  </svelte:fragment>
</Dialog>

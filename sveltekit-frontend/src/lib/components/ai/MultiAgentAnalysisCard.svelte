<script lang="ts">
  import { $derived } from 'svelte';

  interface Props {
    class?: string;
    children?: import('svelte').Snippet;
  }
  import { Card, CardHeader, CardTitle, CardContent } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';
  import { Button } from '$lib/components/ui/button';
  import { Separator } from '$lib/components/ui/separator/Separator.svelte';
  
  let { analysisData = $bindable() } = $props(); // {
    evidenceAnalysis?: unknown;
    personsData?: unknown;
    caseSynthesis?: unknown;
    caseId?: string;
    timestamp?: string;
  };

  // Extract data with fallbacks
  let evidence = $derived(analysisData?.evidenceAnalysis ?? {});
  let persons = $derived(analysisData?.personsData?.persons ?? []);
  let relationships = $derived(analysisData?.personsData?.relationships ?? []);
  let synthesis = $derived(analysisData?.caseSynthesis ?? {});
  
  // Case strength styling
  let strengthColor = $derived({
    strong: 'text-green-600 bg-green-50',
    moderate: 'text-yellow-600 bg-yellow-50', 
    weak: 'text-red-600 bg-red-50'
  }[synthesis.caseStrength] ?? 'text-gray-600 bg-gray-50');

  // Role colors for persons
  const roleColors = {
    suspect: 'bg-red-100 text-red-800',
    witness: 'bg-blue-100 text-blue-800',
    victim: 'bg-purple-100 text-purple-800',
    associate: 'bg-orange-100 text-orange-800',
    unknown: 'bg-gray-100 text-gray-800'
  };
let showDetails = $state(false);
</script>

<Card class="w-full max-w-4xl">
  <CardHeader>
    <div class="flex items-center justify-between">
      <CardTitle class="text-xl font-semibold">
        Multi-Agent Evidence Analysis
      </CardTitle>
      {#if synthesis.caseStrength}
        <Badge class="px-3 py-1 font-medium {strengthColor}">
          Case Strength: {synthesis.caseStrength?.toUpperCase()}
        </Badge>
      {/if}
    </div>
    {#if analysisData.caseId}
      <p class="text-sm text-gray-600">
        Case: {analysisData.caseId} • {analysisData.timestamp ?? 'Recently analyzed'}
      </p>
    {/if}
  </CardHeader>

  <CardContent class="space-y-6">
    <!-- Evidence Summary -->
    {#if evidence.documentType}
      <div>
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
          📄 Evidence Analysis
          <Badge variant="outline">{evidence.documentType}</Badge>
        </h3>
        
        {#if evidence.keyFacts?.length}
          <div class="mb-4">
            <h4 class="font-medium mb-2">Key Facts:</h4>
            <ul class="space-y-1">
              {#each evidence.keyFacts.slice(0, 3) as fact}
                <li class="text-sm text-gray-700 flex items-start gap-2">
                  <span class="text-blue-500 mt-1">•</span>
                  {fact}
                </li>
              {/each}
              {#if evidence.keyFacts.length > 3}
                <li class="text-sm text-gray-500 ml-4">
                  +{evidence.keyFacts.length - 3} more facts
                </li>
              {/if}
            </ul>
          </div>
        {/if}

        {#if evidence.concerns?.length}
          <div class="mb-4">
            <h4 class="font-medium mb-2 text-red-700">⚠️ Concerns:</h4>
            <ul class="space-y-1">
              {#each evidence.concerns as concern}
                <li class="text-sm text-red-600 bg-red-50 p-2 rounded border-l-2 border-red-300">
                  {concern}
                </li>
              {/each}
            </ul>
          </div>
        {/if}
      </div>
      <Separator />
    {/if}

    <!-- Persons of Interest -->
    {#if persons.length > 0}
      <div>
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
          👥 Persons of Interest
          <Badge variant="outline">{persons.length} identified</Badge>
        </h3>
        
        <div class="grid gap-3 md:grid-cols-2">
          {#each persons.slice(0, 4) as person}
            <div class="p-3 border rounded-lg bg-gray-50">
              <div class="flex items-center justify-between mb-2">
                <h4 class="font-medium">{person.name}</h4>
                <Badge class="text-xs {roleColors[person.role] ?? roleColors.unknown}">
                  {person.role?.toUpperCase()}
                </Badge>
              </div>
              
              {#if person.details}
                <div class="text-sm text-gray-600 space-y-1">
                  {#if person.details.age}
                    <p>Age: {person.details.age}</p>
                  {/if}
                  {#if person.details.occupation}
                    <p>Occupation: {person.details.occupation}</p>
                  {/if}
                </div>
              {/if}
              
              {#if person.confidence}
                <div class="mt-2 flex items-center gap-2">
                  <div class="flex-1 bg-gray-200 rounded-full h-1.5">
                    <div 
                      class="bg-blue-500 h-1.5 rounded-full" 
                      style="width: {person.confidence * 100}%"
                    ></div>
                  </div>
                  <span class="text-xs text-gray-500">{Math.round(person.confidence * 100)}%</span>
                </div>
              {/if}
            </div>
          {/each}
          
          {#if persons.length > 4}
            <div class="p-3 border rounded-lg bg-gray-50 flex items-center justify-center text-gray-500">
              +{persons.length - 4} more persons
            </div>
          {/if}
        </div>

        {#if relationships.length > 0}
          <div class="mt-4">
            <h4 class="font-medium mb-2">Key Relationships:</h4>
            <div class="space-y-2">
              {#each relationships.slice(0, 3) as rel}
                <div class="text-sm bg-blue-50 p-2 rounded border-l-2 border-blue-300">
                  <span class="font-medium">{rel.person1}</span>
                  <span class="text-blue-600 mx-2">{rel.relationship?.replace('_', ' ')}</span>
                  <span class="font-medium">{rel.person2}</span>
                  {#if rel.context}
                    <p class="text-gray-600 mt-1">{rel.context}</p>
                  {/if}
                </div>
              {/each}
            </div>
          </div>
        {/if}
      </div>
      <Separator />
    {/if}

    <!-- Case Synthesis & Recommendations -->
    {#if synthesis.keyFindings?.length || synthesis.nextSteps?.length}
      <div>
        <h3 class="text-lg font-semibold mb-3">🎯 Prosecutorial Analysis</h3>
        
        {#if synthesis.keyFindings?.length}
          <div class="mb-4">
            <h4 class="font-medium mb-2">Key Findings:</h4>
            <ul class="space-y-2">
              {#each synthesis.keyFindings.slice(0, 3) as finding}
                <li class="text-sm bg-green-50 p-2 rounded border-l-2 border-green-300">
                  {finding}
                </li>
              {/each}
            </ul>
          </div>
        {/if}

        {#if synthesis.nextSteps?.length}
          <div class="mb-4">
            <h4 class="font-medium mb-2">Next Steps:</h4>
            <ul class="space-y-2">
              {#each synthesis.nextSteps.slice(0, 3) as step}
                <li class="text-sm bg-yellow-50 p-2 rounded border-l-2 border-yellow-300 flex items-start gap-2">
                  <span class="text-yellow-600 mt-1">→</span>
                  {step}
                </li>
              {/each}
            </ul>
          </div>
        {/if}

        {#if synthesis.legalStrategy?.viableCharges?.length}
          <div class="mb-4">
            <h4 class="font-medium mb-2">Viable Charges:</h4>
            <div class="flex flex-wrap gap-2">
              {#each synthesis.legalStrategy.viableCharges as charge}
                <Badge variant="outline" class="bg-purple-50 text-purple-800 border-purple-200">
                  {charge}
                </Badge>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Action Buttons -->
    <div class="flex items-center gap-3 pt-4 border-t">
      <Button 
        variant="outline" 
        size="sm"
        on:on:click={() => showDetails = !showDetails}
      >
        {showDetails ? 'Hide' : 'Show'} Full Analysis
      </Button>
      
      <Button variant="outline" size="sm">
        📊 View Timeline
      </Button>
      
      <Button variant="outline" size="sm">
        🕸️ Relationship Graph  
      </Button>
      
      <Button size="sm">
        📝 Generate Report
      </Button>
    </div>

    <!-- Detailed View -->
    {#if showDetails}
      <div class="mt-6 p-4 bg-gray-50 rounded-lg">
        <h4 class="font-medium mb-3">Detailed Analysis Data</h4>
        <pre class="text-xs overflow-auto max-h-96 bg-white p-3 rounded border">
{JSON.stringify(analysisData, null, 2)}
        </pre>
      </div>
    {/if}
  </CardContent>
</Card>

<style>
  /* Custom scrollbar for JSON display */
  pre::-webkit-scrollbar {
    width: 6px;
    height: 6px;
  }
  
  pre::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
  }
  
  pre::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
  }
  
  pre::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }
</style>
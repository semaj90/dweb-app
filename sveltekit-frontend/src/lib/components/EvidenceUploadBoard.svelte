<!-- Evidence Upload Board Component with AI Analysis -->
<script lang="ts">
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Textarea } from '$lib/components/ui/textarea';
  import { AIAnalysisService } from '$lib/services/ai-analysis';
  
  let files = $state<File[]>([]);
  let analysis = $state('');
  let isAnalyzing = $state(false);
  
  const aiService = new AIAnalysisService();
  
  async function handleFileUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      files = Array.from(input.files);
      await analyzeEvidence();
    }
  }
  
  async function analyzeEvidence() {
    if (!files.length) return;
    
    isAnalyzing = true;
    try {
      const content = await files[0].text();
      analysis = await aiService.analyzeEvidence(content, 'current-case');
    } catch (error) {
      analysis = 'Analysis failed: ' + error.message;
    } finally {
      isAnalyzing = false;
    }
  }
</script>

<div class="p-6 space-y-6">
  <Card>
    <CardHeader>
      <CardTitle>Evidence Upload & AI Analysis</CardTitle>
    </CardHeader>
    <CardContent class="space-y-4">
      <input 
        type="file" 
        onchange={handleFileUpload}
        accept=".txt,.pdf,.doc,.docx"
        class="block w-full text-sm"
      />
      
      {#if files.length > 0}
        <div class="text-sm text-gray-600">
          Uploaded: {files[0].name}
        </div>
      {/if}
      
      {#if isAnalyzing}
        <div class="text-center p-4">
          <div class="animate-spin h-6 w-6 border-b-2 border-blue-600 mx-auto"></div>
          <p class="mt-2">AI analyzing evidence...</p>
        </div>
      {/if}
      
      {#if analysis}
        <div class="border rounded-lg p-4 bg-gray-50">
          <h3 class="font-semibold mb-2">AI Analysis:</h3>
          <Textarea readonly value={analysis} rows={10} />
        </div>
      {/if}
    </CardContent>
  </Card>
</div>

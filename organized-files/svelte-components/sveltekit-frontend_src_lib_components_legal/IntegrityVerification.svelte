<!--
Integrity Verification Component
Displays detailed integrity verification results with AI analysis
-->
<script lang="ts">
  interface Props {
    integrityStatus: 'pending' | 'verified' | 'compromised' | 'requires-attention';
    verificationResults: {;
    originalHash: string
    currentHash: string | undefined;
    aiAnalysis: {;
    showDetails: boolean
  }
  let {
    integrityStatus,
    verificationResults,
    originalHash,
    currentHash,
    aiAnalysis,
    showDetails = false
  } = $props();



  import { Badge } from '$lib/components/ui/badge';
  import { Progress } from '$lib/components/ui/progress';
  import { CheckCircle, XCircle, AlertTriangle, Shield, Hash, Clock, Brain } from 'lucide-svelte';

        hashMatch: boolean
    metadataIntact: boolean
    timestampValid: boolean
    digitalSignatureValid: boolean
    aiAnalysisScore: number
    riskAssessment: string
  } | undefined;
          authenticity: number
    completeness: number
    relevance: number
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
    recommendations: string[];
    flaggedAnomalies: string[];
  } | undefined;
  
  function getStatusIcon(status: string) {
    switch (status) {
      case 'verified':
        return CheckCircle;
      case 'compromised':
        return XCircle;
      case 'requires-attention':
        return AlertTriangle;
      case 'pending':
        return Clock;
      default:
        return Shield;
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'verified':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'compromised':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'requires-attention':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'pending':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  }

  function getRiskLevelColor(riskLevel: string) {
    switch (riskLevel) {
      case 'low':
        return 'text-green-600';
      case 'medium':
        return 'text-yellow-600';
      case 'high':
        return 'text-orange-600';
      case 'critical':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  }

  function getVerificationIcon(passed: boolean) {
    return passed ? CheckCircle: XCircle
  }

  function getVerificationColor(passed: boolean) {
    return passed ? 'text-green-600' : 'text-red-600';
  }

  function formatHash(hash: string) {
    if (!hash) return 'N/A';
    return `${hash.substring(0, 8)}...${hash.substring(hash.length - 8)}`;
  }

  function getOverallScore() {
    if (!verificationResults && !aiAnalysis) return 0;
    
    let score = 0;
    let factors = 0;
    
    if (verificationResults) {
      score += verificationResults.hashMatch ? 25 : 0;
      score += verificationResults.metadataIntact ? 15 : 0;
      score += verificationResults.timestampValid ? 10 : 0;
      score += verificationResults.digitalSignatureValid ? 10 : 0;
      factors += 60;
    }
    
    if (aiAnalysis) {
      score += aiAnalysis.authenticity * 15;
      score += aiAnalysis.completeness * 15;
      score += aiAnalysis.relevance * 10;
      factors += 40;
    }
    
    return Math.round((score / factors) * 100);
  }
</script>

<div class="integrity-verification space-y-6">
  <!-- Overall Status -->
  <div class={`rounded-lg border p-4 ${getStatusColor(integrityStatus)}`}>
    <div class="flex items-center space-x-3">
      <svelte:component this={getStatusIcon(integrityStatus)} class="w-6 h-6" />
      <div class="flex-1">
        <h3 class="font-semibold text-lg">
          Integrity Status: {integrityStatus.toUpperCase().replace('-', ' ')}
        </h3>
        <p class="text-sm opacity-90">
          Overall verification score: {getOverallScore()}%
        </p>
      </div>
      <div class="text-right">
        <div class="text-2xl font-bold">
          {getOverallScore()}%
        </div>
      </div>
    </div>
    
    <!-- Overall Progress Bar -->
    <div class="mt-3">
      <Progress value={getOverallScore()} class="h-2" />
    </div>
  </div>

  <!-- Hash Verification -->
  <div class="bg-white border border-gray-200 rounded-lg p-4">
    <div class="flex items-center space-x-3 mb-4">
      <Hash class="w-5 h-5 text-gray-600" />
      <h4 class="font-semibold">Hash Verification</h4>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Original Hash</label>
        <div class="font-mono text-sm bg-gray-50 p-2 rounded border">
          {formatHash(originalHash)}
        </div>
      </div>
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Current Hash</label>
        <div class="font-mono text-sm bg-gray-50 p-2 rounded border">
          {currentHash ? formatHash(currentHash) : 'Computing...'}
        </div>
      </div>
    </div>
    
    {#if verificationResults}
      <div class="mt-4 flex items-center space-x-2">
        <svelte:component 
          this={getVerificationIcon(verificationResults.hashMatch)} 
          class={`w-5 h-5 ${getVerificationColor(verificationResults.hashMatch)}`} 
        />
        <span class={`text-sm font-medium ${getVerificationColor(verificationResults.hashMatch)}`}>
          {verificationResults.hashMatch ? 'Hashes match - integrity verified' : 'Hash mismatch detected - integrity compromised'}
        </span>
      </div>
    {/if}
  </div>

  <!-- Detailed Verification Results -->
  {#if verificationResults && showDetails}
    <div class="bg-white border border-gray-200 rounded-lg p-4">
      <div class="flex items-center space-x-3 mb-4">
        <Shield class="w-5 h-5 text-gray-600" />
        <h4 class="font-semibold">Detailed Verification</h4>
      </div>
      
      <div class="space-y-3">
        <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
          <span class="text-sm font-medium">Hash Integrity</span>
          <div class="flex items-center space-x-2">
            <svelte:component 
              this={getVerificationIcon(verificationResults.hashMatch)} 
              class={`w-4 h-4 ${getVerificationColor(verificationResults.hashMatch)}`} 
            />
            <Badge variant={verificationResults.hashMatch ? 'success' : 'destructive'}>
              {verificationResults.hashMatch ? 'Pass' : 'Fail'}
            </Badge>
          </div>
        </div>
        
        <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
          <span class="text-sm font-medium">Metadata Integrity</span>
          <div class="flex items-center space-x-2">
            <svelte:component 
              this={getVerificationIcon(verificationResults.metadataIntact)} 
              class={`w-4 h-4 ${getVerificationColor(verificationResults.metadataIntact)}`} 
            />
            <Badge variant={verificationResults.metadataIntact ? 'success' : 'destructive'}>
              {verificationResults.metadataIntact ? 'Pass' : 'Fail'}
            </Badge>
          </div>
        </div>
        
        <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
          <span class="text-sm font-medium">Timestamp Validation</span>
          <div class="flex items-center space-x-2">
            <svelte:component 
              this={getVerificationIcon(verificationResults.timestampValid)} 
              class={`w-4 h-4 ${getVerificationColor(verificationResults.timestampValid)}`} 
            />
            <Badge variant={verificationResults.timestampValid ? 'success' : 'destructive'}>
              {verificationResults.timestampValid ? 'Pass' : 'Fail'}
            </Badge>
          </div>
        </div>
        
        <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
          <span class="text-sm font-medium">Digital Signature</span>
          <div class="flex items-center space-x-2">
            <svelte:component 
              this={getVerificationIcon(verificationResults.digitalSignatureValid)} 
              class={`w-4 h-4 ${getVerificationColor(verificationResults.digitalSignatureValid)}`} 
            />
            <Badge variant={verificationResults.digitalSignatureValid ? 'success' : 'destructive'}>
              {verificationResults.digitalSignatureValid ? 'Pass' : 'Fail'}
            </Badge>
          </div>
        </div>
        
        <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
          <span class="text-sm font-medium">AI Analysis Score</span>
          <div class="flex items-center space-x-2">
            <div class="text-sm font-semibold">
              {Math.round(verificationResults.aiAnalysisScore * 100)}%
            </div>
            <Badge variant={verificationResults.aiAnalysisScore > 0.7 ? 'success' : 'warning'}>
              {verificationResults.aiAnalysisScore > 0.7 ? 'Good' : 'Review'}
            </Badge>
          </div>
        </div>
      </div>
      
      {#if verificationResults.riskAssessment}
        <div class="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
          <p class="text-sm">
            <strong>Risk Assessment:</strong> {verificationResults.riskAssessment}
          </p>
        </div>
      {/if}
    </div>
  {/if}

  <!-- AI Analysis Results -->
  {#if aiAnalysis && showDetails}
    <div class="bg-white border border-gray-200 rounded-lg p-4">
      <div class="flex items-center space-x-3 mb-4">
        <Brain class="w-5 h-5 text-gray-600" />
        <h4 class="font-semibold">AI Analysis Results</h4>
      </div>
      
      <!-- AI Scores -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div class="text-center">
          <div class="text-2xl font-bold text-blue-600">
            {Math.round(aiAnalysis.authenticity * 100)}%
          </div>
          <div class="text-sm text-gray-600">Authenticity</div>
          <Progress value={aiAnalysis.authenticity * 100} class="mt-2 h-2" />
        </div>
        
        <div class="text-center">
          <div class="text-2xl font-bold text-green-600">
            {Math.round(aiAnalysis.completeness * 100)}%
          </div>
          <div class="text-sm text-gray-600">Completeness</div>
          <Progress value={aiAnalysis.completeness * 100} class="mt-2 h-2" />
        </div>
        
        <div class="text-center">
          <div class="text-2xl font-bold text-purple-600">
            {Math.round(aiAnalysis.relevance * 100)}%
          </div>
          <div class="text-sm text-gray-600">Relevance</div>
          <Progress value={aiAnalysis.relevance * 100} class="mt-2 h-2" />
        </div>
      </div>
      
      <!-- Risk Level -->
      <div class="mb-4">
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">Overall Risk Level</span>
          <Badge variant="outline" class={getRiskLevelColor(aiAnalysis.riskLevel)}>
            {aiAnalysis.riskLevel.toUpperCase()}
          </Badge>
        </div>
      </div>
      
      <!-- Recommendations -->
      {#if aiAnalysis.recommendations.length > 0}
        <div class="mb-4">
          <h5 class="font-medium mb-2">AI Recommendations</h5>
          <ul class="space-y-2">
            {#each aiAnalysis.recommendations as recommendation}
              <li class="flex items-start space-x-2">
                <CheckCircle class="w-4 h-4 text-green-600 mt-0.5 shrink-0" />
                <span class="text-sm text-gray-700">{recommendation}</span>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
      
      <!-- Flagged Anomalies -->
      {#if aiAnalysis.flaggedAnomalies.length > 0}
        <div>
          <h5 class="font-medium mb-2 text-orange-600">Flagged Anomalies</h5>
          <ul class="space-y-2">
            {#each aiAnalysis.flaggedAnomalies as anomaly}
              <li class="flex items-start space-x-2">
                <AlertTriangle class="w-4 h-4 text-orange-600 mt-0.5 shrink-0" />
                <span class="text-sm text-gray-700">{anomaly}</span>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .integrity-verification {
    animation: fadeIn 0.5s ease-out;
  }
  
  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>
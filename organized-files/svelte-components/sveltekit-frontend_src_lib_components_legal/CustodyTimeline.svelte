<!--
Custody Timeline Component
Displays the chronological chain of custody events with detailed audit trail
-->
<script lang="ts">
  interface Props {
    events: Array<{;
    currentStage: string
  }
  let {
    events,
    currentStage
  } = $props();



  import { Badge } from '$lib/components/ui/badge';
  import { CheckCircle, AlertTriangle, Clock, FileCheck, Users, Shield, UserCheck } from 'lucide-svelte';

      id: string
    eventType: 'intake' | 'transfer' | 'verification' | 'analysis' | 'approval' | 'finalization';
    timestamp: string
    userId: string
    details: Record<string, any>;
    signature?: string;
  }>;
  
  function getEventIcon(eventType: string) {
    switch (eventType) {
      case 'intake':
        return Shield;
      case 'transfer':
        return Users;
      case 'verification':
        return FileCheck;
      case 'analysis':
        return CheckCircle;
      case 'approval':
        return UserCheck;
      case 'finalization':
        return CheckCircle;
      default:
        return Clock;
    }
  }

  function getEventColor(eventType: string) {
    switch (eventType) {
      case 'intake':
        return 'bg-blue-100 text-blue-800';
      case 'transfer':
        return 'bg-purple-100 text-purple-800';
      case 'verification':
        return 'bg-green-100 text-green-800';
      case 'analysis':
        return 'bg-indigo-100 text-indigo-800';
      case 'approval':
        return 'bg-emerald-100 text-emerald-800';
      case 'finalization':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  function formatEventTitle(eventType: string) {
    return eventType.split('-').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  }

  function formatTimestamp(timestamp: string) {
    return new Date(timestamp).toLocaleString();
  }

  function getEventDetails(event: any) {
    switch (event.eventType) {
      case 'intake':
        return {
          primary: `Evidence taken into custody`,
          secondary: `Hash verified: ${event.details.hashMatch ? 'Yes' : 'No'}`,
          extra: event.details.originalHash ? `Hash: ${event.details.originalHash.substring(0, 8)}...` : ''
        };
      case 'transfer':
        return {
          primary: `Custody transferred`,
          secondary: `From: ${event.details.fromCustodian} → To: ${event.details.toCustodian}`,
          extra: `Reason: ${event.details.transferReason}`
        };
      case 'verification':
        return {
          primary: `Integrity verification completed`,
          secondary: `Status: ${event.details.integrityStatus}`,
          extra: event.details.verificationResults ? 
            `AI Score: ${(event.details.verificationResults.aiAnalysisScore * 100).toFixed(0)}%` : ''
        };
      case 'analysis':
        return {
          primary: `AI analysis completed`,
          secondary: `Risk Level: ${event.details.aiAnalysis?.riskLevel || 'Unknown'}`,
          extra: event.details.aiAnalysis ? 
            `Models: ${event.details.models?.join(', ') || 'Multiple'}` : ''
        };
      case 'approval':
        return {
          primary: `Custody approved`,
          secondary: `Approval status: ${event.details.approvalStatus}`,
          extra: event.details.finalIntegrityStatus ? 
            `Final status: ${event.details.finalIntegrityStatus}` : ''
        };
      case 'finalization':
        return {
          primary: `Custody workflow finalized`,
          secondary: `Total events: ${event.details.custodyReport?.totalEvents || 0}`,
          extra: event.details.custodyReport?.totalProcessingTime ? 
            `Duration: ${Math.round(event.details.custodyReport.totalProcessingTime / 1000)}s` : ''
        };
      default:
        return {
          primary: formatEventTitle(event.eventType),
          secondary: 'Event processed',
          extra: ''
        };
    }
  }
</script>

<div class="custody-timeline space-y-4">
  {#if events.length === 0}
    <div class="text-center py-8 text-gray-500">
      <Clock class="w-12 h-12 mx-auto mb-4 opacity-50" />
      <p>No custody events recorded yet</p>
    </div>
  {:else}
    <div class="relative">
      <!-- Timeline line -->
      <div class="absolute left-6 top-0 bottom-0 w-px bg-gray-200"></div>
      
      {#each events as event, index}
        {@const details = getEventDetails(event)}
        {@const isLast = index === events.length - 1}
        
        <div class="relative flex items-start space-x-4 pb-6">
          <!-- Timeline dot -->
          <div class={`
            relative z-10 flex items-center justify-center w-12 h-12 rounded-full border-4 border-white shadow-lg
            ${getEventColor(event.eventType)}
          `}>
            <svelte:component this={getEventIcon(event.eventType)} class="w-5 h-5" />
          </div>
          
          <!-- Event content -->
          <div class="flex-1 min-w-0 bg-white border border-gray-200 rounded-lg shadow-sm p-4">
            <div class="flex items-start justify-between mb-2">
              <div class="flex-1">
                <h4 class="font-semibold text-gray-900 mb-1">
                  {details.primary}
                </h4>
                <p class="text-sm text-gray-600 mb-2">
                  {details.secondary}
                </p>
                {#if details.extra}
                  <p class="text-xs text-gray-500">
                    {details.extra}
                  </p>
                {/if}
              </div>
              
              <!-- Event badge -->
              <Badge variant="secondary" class="ml-4 shrink-0">
                {formatEventTitle(event.eventType)}
              </Badge>
            </div>
            
            <!-- Event metadata -->
            <div class="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-100">
              <span>
                User: {event.userId}
              </span>
              <span>
                {formatTimestamp(event.timestamp)}
              </span>
            </div>
            
            <!-- Digital signature indicator -->
            {#if event.signature}
              <div class="mt-2 pt-2 border-t border-gray-100">
                <div class="flex items-center text-xs text-green-600">
                  <CheckCircle class="w-3 h-3 mr-1" />
                  Digitally signed: {event.signature.substring(0, 16)}...
                </div>
              </div>
            {/if}
            
            <!-- Detailed information (expandable) -->
            {#if event.details && Object.keys(event.details).length > 0}
              <details class="mt-2 pt-2 border-t border-gray-100">
                <summary class="cursor-pointer text-xs text-blue-600 hover:text-blue-800">
                  View detailed information
                </summary>
                <div class="mt-2 p-2 bg-gray-50 rounded text-xs">
                  <pre class="whitespace-pre-wrap text-gray-700 font-mono text-xs overflow-auto max-h-32">
{JSON.stringify(event.details, null, 2)}
                  </pre>
                </div>
              </details>
            {/if}
          </div>
        </div>
      {/each}
      
      <!-- Current stage indicator (if workflow is active) -->
      {#if currentStage && !['completed', 'failed', 'cancelled'].includes(currentStage)}
        <div class="relative flex items-start space-x-4">
          <!-- Active stage dot -->
          <div class="relative z-10 flex items-center justify-center w-12 h-12 rounded-full border-4 border-white shadow-lg bg-blue-100 text-blue-800 animate-pulse">
            <Clock class="w-5 h-5" />
          </div>
          
          <!-- Active stage content -->
          <div class="flex-1 min-w-0 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div class="flex items-center justify-between mb-2">
              <h4 class="font-semibold text-blue-900">
                Currently: {formatEventTitle(currentStage)}
              </h4>
              <Badge variant="secondary" class="bg-blue-100 text-blue-800">
                In Progress
              </Badge>
            </div>
            <p class="text-sm text-blue-700">
              This stage is currently being processed...
            </p>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .custody-timeline {
    max-height: 600px;
    overflow-y: auto
  }
  
  /* Smooth scrolling for timeline */
  .custody-timeline {
    scroll-behavior: smooth
  }
  
  /* Custom scrollbar */
  .custody-timeline::-webkit-scrollbar {
    width: 6px;
  }
  
  .custody-timeline::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
  }
  
  .custody-timeline::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
  }
  
  .custody-timeline::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }
</style>
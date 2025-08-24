<!-- Chain of Custody Tracker for Legal AI App -->
<script lang="ts">
  import { Shield, User, Calendar, MapPin, FileCheck, AlertTriangle, Lock, Unlock } from 'lucide-svelte';
  import { cn } from '$lib/utils';
  import { $props, $derived } from 'svelte';

  export interface CustodyTransfer {
    id: string;
    timestamp: Date;
    fromPerson: string;
    toPerson: string;
    location: string;
    reason: string;
    condition: 'excellent' | 'good' | 'fair' | 'poor' | 'damaged';
    photos?: string[];
    signature: string;
    witnessSignature?: string;
    notes?: string;
    verified: boolean;
  }

  export interface EvidenceItem {
    id: string;
    itemNumber: string;
    description: string;
    category: 'physical' | 'digital' | 'document' | 'biological' | 'chemical' | 'other';
    collectedDate: Date;
    collectedBy: string;
    currentCustodian: string;
    location: string;
    condition: 'excellent' | 'good' | 'fair' | 'poor' | 'damaged';
    sealed: boolean;
    chainOfCustody: CustodyTransfer[];
    compromised: boolean;
    compromisedReason?: string;
  }

  export interface ChainOfCustodyProps {
    evidence: EvidenceItem;
    showFullHistory?: boolean;
    interactive?: boolean;
    onTransferEvidence?: (evidence: EvidenceItem) => void;
    onViewDetails?: (transfer: CustodyTransfer) => void;
    class?: string;
  }

  let {
    evidence,
    showFullHistory = true,
    interactive = true,
    onTransferEvidence,
    onViewDetails,
    class: className = ''
  }: ChainOfCustodyProps = $props();

  // Sort transfers by date (newest first)
  const sortedTransfers = $derived(() => {
    return [...evidence.chainOfCustody].sort((a, b) => 
      b.timestamp.getTime() - a.timestamp.getTime()
    );
  });

  // Get the most recent transfer
  const latestTransfer = $derived(() => sortedTransfers[0]);

  // Evidence category configurations
  const categoryConfig = {
    physical: { icon: Shield, color: 'text-blue-400', bg: 'bg-blue-500/10' },
    digital: { icon: FileCheck, color: 'text-green-400', bg: 'bg-green-500/10' },
    document: { icon: FileCheck, color: 'text-yellow-400', bg: 'bg-yellow-500/10' },
    biological: { icon: Shield, color: 'text-red-400', bg: 'bg-red-500/10' },
    chemical: { icon: Shield, color: 'text-purple-400', bg: 'bg-purple-500/10' },
    other: { icon: Shield, color: 'text-gray-400', bg: 'bg-gray-500/10' }
  };

  // Condition configurations
  const conditionConfig = {
    excellent: { label: 'Excellent', class: 'bg-green-500/20 text-green-400' },
    good: { label: 'Good', class: 'bg-blue-500/20 text-blue-400' },
    fair: { label: 'Fair', class: 'bg-yellow-500/20 text-yellow-400' },
    poor: { label: 'Poor', class: 'bg-orange-500/20 text-orange-400' },
    damaged: { label: 'Damaged', class: 'bg-red-500/20 text-red-400' }
  };

  function formatDateTime(date: Date): string {
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  function getTimeSince(date: Date): string {
    const now = new Date();
    const diffTime = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor(diffTime / (1000 * 60 * 60));
    
    if (diffDays > 0) {
      return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
    } else if (diffHours > 0) {
      return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
    } else {
      return 'Less than an hour ago';
    }
  }
</script>

<div class={cn('chain-of-custody w-full space-y-4', className)}>
  <!-- Evidence Header -->
  <div class={cn(
    'bg-yorha-bg-secondary border rounded-lg p-4',
    evidence.compromised ? 'border-red-500/30 bg-red-500/5' : 'border-yorha-border'
  )}>
    <div class="flex items-start justify-between mb-3">
      <div class="flex items-center gap-3">
        {@const config = categoryConfig[evidence.category]}
        {@const IconComponent = config.icon}
        
        <div class={cn('p-2 rounded-md', config.bg)}>
          <IconComponent class={cn('w-5 h-5', config.color)} />
        </div>
        
        <div>
          <h3 class="text-lg font-semibold text-yorha-text-primary font-mono">
            Evidence #{evidence.itemNumber}
          </h3>
          <p class="text-sm text-yorha-text-secondary font-mono">
            {evidence.description}
          </p>
        </div>
      </div>

      <!-- Evidence Status -->
      <div class="flex items-center gap-2">
        {#if evidence.sealed}
          <div class="flex items-center gap-1 text-green-400">
            <Lock class="w-4 h-4" />
            <span class="text-xs font-mono">SEALED</span>
          </div>
        {:else}
          <div class="flex items-center gap-1 text-orange-400">
            <Unlock class="w-4 h-4" />
            <span class="text-xs font-mono">UNSEALED</span>
          </div>
        {/if}

        {#if evidence.compromised}
          <div class="flex items-center gap-1 text-red-400">
            <AlertTriangle class="w-4 h-4" />
            <span class="text-xs font-mono">COMPROMISED</span>
          </div>
        {/if}
      </div>
    </div>

    <!-- Current Status Grid -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm font-mono">
      <div>
        <span class="text-yorha-text-secondary">Current Custodian:</span>
        <div class="text-yorha-text-primary font-medium">{evidence.currentCustodian}</div>
      </div>
      <div>
        <span class="text-yorha-text-secondary">Location:</span>
        <div class="text-yorha-text-primary font-medium">{evidence.location}</div>
      </div>
      <div>
        <span class="text-yorha-text-secondary">Condition:</span>
        <span class={cn(
          'inline-block px-2 py-0.5 rounded text-xs',
          conditionConfig[evidence.condition].class
        )}>
          {conditionConfig[evidence.condition].label}
        </span>
      </div>
      <div>
        <span class="text-yorha-text-secondary">Category:</span>
        <div class="text-yorha-text-primary font-medium capitalize">{evidence.category}</div>
      </div>
    </div>

    <!-- Compromised Warning -->
    {#if evidence.compromised}
      <div class="mt-3 p-3 bg-red-500/10 border border-red-500/20 rounded text-sm font-mono">
        <div class="flex items-center gap-2 text-red-400 font-medium mb-1">
          <AlertTriangle class="w-4 h-4" />
          Chain of Custody Compromised
        </div>
        <p class="text-red-300">
          {evidence.compromisedReason || 'Reason not specified'}
        </p>
      </div>
    {/if}

    <!-- Transfer Action -->
    {#if interactive && onTransferEvidence}
      <div class="mt-3 flex justify-end">
        <button
          onclick={() => onTransferEvidence?.(evidence)}
          class="px-4 py-2 text-sm font-mono bg-yorha-primary/10 text-yorha-primary border border-yorha-primary/20 rounded hover:bg-yorha-primary/20 transition-colors"
        >
          Transfer Custody
        </button>
      </div>
    {/if}
  </div>

  <!-- Chain of Custody History -->
  <div class="bg-yorha-bg-secondary border border-yorha-border rounded-lg p-4">
    <div class="flex items-center justify-between mb-4">
      <h4 class="text-base font-semibold text-yorha-text-primary font-mono">
        Chain of Custody History
      </h4>
      <span class="text-sm text-yorha-text-secondary font-mono">
        {evidence.chainOfCustody.length} transfer{evidence.chainOfCustody.length !== 1 ? 's' : ''}
      </span>
    </div>

    {#if evidence.chainOfCustody.length === 0}
      <div class="text-center py-8 text-yorha-text-secondary font-mono">
        <Shield class="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>No custody transfers recorded</p>
      </div>
    {:else}
      <!-- Timeline -->
      <div class="relative">
        <!-- Timeline Line -->
        <div class="absolute left-6 top-0 bottom-0 w-px bg-yorha-border"></div>

        <div class="space-y-4">
          {#each showFullHistory ? sortedTransfers : sortedTransfers.slice(0, 3) as transfer, index (transfer.id)}
            <div 
              class={cn(
                'relative flex items-start gap-4',
                interactive && 'cursor-pointer group'
              )}
              onclick={() => interactive && onViewDetails?.(transfer)}
            >
              <!-- Timeline Node -->
              <div class={cn(
                'relative z-10 flex items-center justify-center w-12 h-12 rounded-full border-2',
                transfer.verified ? 'bg-green-500/10 border-green-500/30' : 'bg-yellow-500/10 border-yellow-500/30',
                interactive && 'group-hover:scale-110 transition-transform'
              )}>
                {#if transfer.verified}
                  <FileCheck class="w-5 h-5 text-green-400" />
                {:else}
                  <AlertTriangle class="w-5 h-5 text-yellow-400" />
                {/if}
              </div>

              <!-- Transfer Content -->
              <div class="flex-1 min-w-0">
                <div class={cn(
                  'bg-yorha-bg-tertiary border border-yorha-border rounded-lg p-4',
                  interactive && 'group-hover:border-yorha-primary/30 transition-colors'
                )}>
                  <!-- Transfer Header -->
                  <div class="flex items-start justify-between mb-3">
                    <div>
                      <div class="flex items-center gap-2 text-sm font-mono">
                        <User class="w-4 h-4 text-yorha-text-secondary" />
                        <span class="text-yorha-text-primary">{transfer.fromPerson}</span>
                        <span class="text-yorha-text-secondary">→</span>
                        <span class="text-yorha-text-primary font-medium">{transfer.toPerson}</span>
                      </div>
                      <div class="flex items-center gap-2 text-xs text-yorha-text-secondary font-mono mt-1">
                        <Calendar class="w-3 h-3" />
                        {formatDateTime(transfer.timestamp)}
                        <span>•</span>
                        <span>{getTimeSince(transfer.timestamp)}</span>
                      </div>
                    </div>

                    <!-- Verification Status -->
                    <span class={cn(
                      'px-2 py-1 text-xs font-mono rounded border',
                      transfer.verified 
                        ? 'bg-green-500/20 text-green-400 border-green-500/30'
                        : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                    )}>
                      {transfer.verified ? 'VERIFIED' : 'UNVERIFIED'}
                    </span>
                  </div>

                  <!-- Transfer Details -->
                  <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
                    <div>
                      <span class="text-yorha-text-secondary">Reason:</span>
                      <div class="text-yorha-text-primary">{transfer.reason}</div>
                    </div>
                    <div>
                      <span class="text-yorha-text-secondary">Location:</span>
                      <div class="flex items-center gap-1 text-yorha-text-primary">
                        <MapPin class="w-3 h-3" />
                        {transfer.location}
                      </div>
                    </div>
                    <div>
                      <span class="text-yorha-text-secondary">Condition:</span>
                      <span class={cn(
                        'inline-block px-1.5 py-0.5 rounded',
                        conditionConfig[transfer.condition].class
                      )}>
                        {conditionConfig[transfer.condition].label}
                      </span>
                    </div>
                    <div>
                      <span class="text-yorha-text-secondary">Signatures:</span>
                      <div class="text-yorha-text-primary">
                        {transfer.signature}
                        {#if transfer.witnessSignature}
                          <br><span class="text-yorha-text-secondary">Witness:</span> {transfer.witnessSignature}
                        {/if}
                      </div>
                    </div>
                  </div>

                  <!-- Notes -->
                  {#if transfer.notes}
                    <div class="mt-3 p-2 bg-yorha-bg-primary rounded text-xs font-mono">
                      <span class="text-yorha-text-secondary">Notes:</span>
                      <div class="text-yorha-text-primary mt-1">{transfer.notes}</div>
                    </div>
                  {/if}
                </div>
              </div>
            </div>
          {/each}
        </div>

        <!-- Show More Button -->
        {#if !showFullHistory && sortedTransfers.length > 3}
          <div class="text-center mt-4">
            <button 
              onclick={() => showFullHistory = true}
              class="text-sm font-mono text-yorha-primary hover:text-yorha-accent transition-colors"
            >
              Show {sortedTransfers.length - 3} more transfer{sortedTransfers.length - 3 !== 1 ? 's' : ''}
            </button>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .chain-of-custody {
    --custody-line-color: rgb(var(--yorha-border));
  }
</style>
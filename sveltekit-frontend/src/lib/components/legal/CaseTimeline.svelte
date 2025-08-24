<!-- Case Timeline Component for Legal AI App -->
<script lang="ts">
  import { Calendar, FileText, Users, Scale, AlertCircle, CheckCircle } from 'lucide-svelte';
  import { cn } from '$lib/utils';
  import { $props, $derived } from 'svelte';

  export interface TimelineEvent {
    id: string;
    date: Date;
    title: string;
    description?: string;
    type: 'filing' | 'hearing' | 'evidence' | 'meeting' | 'deadline' | 'decision' | 'milestone';
    status: 'completed' | 'pending' | 'overdue' | 'cancelled';
    participants?: string[];
    documents?: string[];
    location?: string;
    priority?: 'low' | 'medium' | 'high' | 'critical';
    metadata?: Record<string, any>;
  }

  export interface CaseTimelineProps {
    caseId: string;
    caseName: string;
    events: TimelineEvent[];
    showFutureEvents?: boolean;
    compactMode?: boolean;
    interactive?: boolean;
    onEventClick?: (event: TimelineEvent) => void;
    onAddEvent?: () => void;
    class?: string;
  }

  let {
    caseId,
    caseName,
    events = [],
    showFutureEvents = true,
    compactMode = false,
    interactive = true,
    onEventClick,
    onAddEvent,
    class: className = ''
  }: CaseTimelineProps = $props();

  // Sort events by date
  const sortedEvents = $derived(() => {
    const now = new Date();
    const filtered = showFutureEvents 
      ? events 
      : events.filter(event => event.date <= now);
    
    return filtered.sort((a, b) => b.date.getTime() - a.date.getTime());
  });

  // Event type configurations
  const eventConfig = {
    filing: { icon: FileText, color: 'text-blue-400', bg: 'bg-blue-500/10', border: 'border-blue-500/20' },
    hearing: { icon: Scale, color: 'text-purple-400', bg: 'bg-purple-500/10', border: 'border-purple-500/20' },
    evidence: { icon: FileText, color: 'text-green-400', bg: 'bg-green-500/10', border: 'border-green-500/20' },
    meeting: { icon: Users, color: 'text-yellow-400', bg: 'bg-yellow-500/10', border: 'border-yellow-500/20' },
    deadline: { icon: AlertCircle, color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/20' },
    decision: { icon: CheckCircle, color: 'text-yorha-primary', bg: 'bg-yorha-primary/10', border: 'border-yorha-primary/20' },
    milestone: { icon: Calendar, color: 'text-yorha-accent', bg: 'bg-yorha-accent/10', border: 'border-yorha-accent/20' }
  };

  // Status configurations
  const statusConfig = {
    completed: { label: 'Completed', class: 'bg-green-500/20 text-green-400 border-green-500/30' },
    pending: { label: 'Pending', class: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' },
    overdue: { label: 'Overdue', class: 'bg-red-500/20 text-red-400 border-red-500/30' },
    cancelled: { label: 'Cancelled', class: 'bg-gray-500/20 text-gray-400 border-gray-500/30' }
  };

  function formatDate(date: Date): string {
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  }

  function formatTime(date: Date): string {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  function isToday(date: Date): boolean {
    const today = new Date();
    return date.toDateString() === today.toDateString();
  }

  function isPast(date: Date): boolean {
    return date < new Date();
  }
</script>

<div class={cn('case-timeline w-full space-y-4', className)}>
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div>
      <h3 class="text-lg font-semibold text-yorha-text-primary font-mono">
        Case Timeline
      </h3>
      <p class="text-sm text-yorha-text-secondary font-mono">
        {caseName} • {sortedEvents.length} events
      </p>
    </div>
    
    {#if onAddEvent && interactive}
      <button
        onclick={onAddEvent}
        class="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium font-mono bg-yorha-primary/10 text-yorha-primary border border-yorha-primary/20 rounded-md hover:bg-yorha-primary/20 transition-colors"
      >
        <Calendar class="w-4 h-4" />
        Add Event
      </button>
    {/if}
  </div>

  <!-- Timeline -->
  <div class="relative">
    {#if sortedEvents.length === 0}
      <div class="text-center py-12 text-yorha-text-secondary font-mono">
        <Calendar class="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>No timeline events recorded</p>
        {#if onAddEvent}
          <button
            onclick={onAddEvent}
            class="mt-2 text-yorha-primary hover:text-yorha-accent transition-colors"
          >
            Add the first event
          </button>
        {/if}
      </div>
    {:else}
      <!-- Timeline Line -->
      <div class="absolute left-6 top-0 bottom-0 w-px bg-yorha-border"></div>

      <div class="space-y-6">
        {#each sortedEvents as event, index (event.id)}
          {@const config = eventConfig[event.type]}
          {@const status = statusConfig[event.status]}
          {@const IconComponent = config.icon}
          
          <div 
            class={cn(
              'relative flex items-start gap-4',
              interactive && 'cursor-pointer group',
              compactMode && 'gap-3'
            )}
            onclick={() => interactive && onEventClick?.(event)}
          >
            <!-- Timeline Node -->
            <div class={cn(
              'relative z-10 flex items-center justify-center w-12 h-12 rounded-full border-2',
              config.bg,
              config.border,
              interactive && 'group-hover:scale-110 transition-transform'
            )}>
              <IconComponent class={cn('w-5 h-5', config.color)} />
            </div>

            <!-- Event Content -->
            <div class={cn(
              'flex-1 min-w-0 pb-6',
              compactMode && 'pb-4'
            )}>
              <div class={cn(
                'bg-yorha-bg-secondary border border-yorha-border rounded-lg p-4',
                interactive && 'group-hover:border-yorha-primary/30 group-hover:bg-yorha-bg-tertiary transition-colors',
                compactMode && 'p-3'
              )}>
                <!-- Event Header -->
                <div class="flex items-start justify-between mb-2">
                  <div class="flex-1">
                    <h4 class={cn(
                      'font-semibold text-yorha-text-primary font-mono',
                      compactMode ? 'text-sm' : 'text-base'
                    )}>
                      {event.title}
                    </h4>
                    <div class="flex items-center gap-3 mt-1">
                      <span class={cn(
                        'font-mono text-yorha-text-secondary',
                        compactMode ? 'text-xs' : 'text-sm',
                        isToday(event.date) && 'text-yorha-accent font-medium'
                      )}>
                        {formatDate(event.date)} • {formatTime(event.date)}
                        {#if isToday(event.date)}
                          <span class="ml-1 text-yorha-accent">TODAY</span>
                        {/if}
                      </span>
                      
                      {#if event.priority && event.priority !== 'medium'}
                        <span class={cn(
                          'px-2 py-0.5 text-xs font-mono rounded',
                          event.priority === 'critical' && 'bg-red-500/20 text-red-400 border border-red-500/30',
                          event.priority === 'high' && 'bg-orange-500/20 text-orange-400 border border-orange-500/30',
                          event.priority === 'low' && 'bg-gray-500/20 text-gray-400 border border-gray-500/30'
                        )}>
                          {event.priority.toUpperCase()}
                        </span>
                      {/if}
                    </div>
                  </div>

                  <!-- Status Badge -->
                  <span class={cn(
                    'px-2 py-1 text-xs font-mono rounded border',
                    status.class
                  )}>
                    {status.label}
                  </span>
                </div>

                <!-- Event Description -->
                {#if event.description && !compactMode}
                  <p class="text-sm text-yorha-text-secondary font-mono mb-3">
                    {event.description}
                  </p>
                {/if}

                <!-- Event Metadata -->
                {#if !compactMode && (event.participants?.length || event.documents?.length || event.location)}
                  <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
                    {#if event.participants?.length}
                      <div>
                        <span class="text-yorha-text-secondary">Participants:</span>
                        <div class="mt-1 space-y-1">
                          {#each event.participants as participant}
                            <div class="text-yorha-text-primary">{participant}</div>
                          {/each}
                        </div>
                      </div>
                    {/if}

                    {#if event.documents?.length}
                      <div>
                        <span class="text-yorha-text-secondary">Documents:</span>
                        <div class="mt-1 space-y-1">
                          {#each event.documents as document}
                            <div class="text-yorha-primary hover:text-yorha-accent cursor-pointer">
                              {document}
                            </div>
                          {/each}
                        </div>
                      </div>
                    {/if}

                    {#if event.location}
                      <div>
                        <span class="text-yorha-text-secondary">Location:</span>
                        <div class="mt-1 text-yorha-text-primary">{event.location}</div>
                      </div>
                    {/if}
                  </div>
                {/if}
              </div>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .case-timeline {
    --timeline-line-color: rgb(var(--yorha-border));
  }
</style>
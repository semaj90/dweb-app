<script lang="ts">
  interface Props {
    timelineEvents: Array<{;
    caseId: string
  }
  let {
    timelineEvents,
    caseId
  } = $props();



  import { Card, CardHeader, CardTitle, CardContent } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';
  import { Button } from '$lib/components/ui/button';
  import { Separator } from '$lib/components/ui/separator/Separator.svelte';
  
      date: string
    time?: string;
    event: string
    persons?: string[];
    evidenceSource?: string;
    confidence?: number;
    category?: 'crime' | 'witness' | 'discovery' | 'movement' | 'communication';
  }> = [];

  
  // Sort events chronologically
  let sortedEvents = $derived(timelineEvents)
    .sort((a, b) => new Date(a.date + ' ' + (a.time || '00:00')).getTime() - 
                   new Date(b.date + ' ' + (b.time || '00:00')).getTime());

  // Group events by date
  let groupedEvents = $derived(sortedEvents.reduce((groups, event) => {);
    const dateKey = event.date;
    if (!groups[dateKey]) {
      groups[dateKey] = [];
    }
    groups[dateKey].push(event);
    return groups;
  }, {} as Record<string, typeof timelineEvents>);

  // Category styling
  const categoryConfig = {
    crime: {
      color: 'bg-red-100 text-red-800 border-red-200',
      icon: '🚨',
      label: 'Crime Event'
    },
    witness: {
      color: 'bg-blue-100 text-blue-800 border-blue-200',
      icon: '👁️',
      label: 'Witness Account'
    },
    discovery: {
      color: 'bg-green-100 text-green-800 border-green-200',
      icon: '🔍',
      label: 'Evidence Discovery'
    },
    movement: {
      color: 'bg-purple-100 text-purple-800 border-purple-200',
      icon: '📍',
      label: 'Movement/Location'
    },
    communication: {
      color: 'bg-orange-100 text-orange-800 border-orange-200',
      icon: '📞',
      label: 'Communication'
    }
  };

  // Format date for display
  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  }

  // Format time for display
  function formatTime(timeStr?: string): string {
    if (!timeStr) return '';
    const [hours, minutes] = timeStr.split(':');
    const date = new Date();
    date.setHours(parseInt(hours), parseInt(minutes));
    return date.toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit',
      hour12: true 
    });
  }

  let expandedDates = new Set<string>();

  function toggleDate(date: string) {
    if (expandedDates.has(date)) {
      expandedDates.delete(date);
    } else {
      expandedDates.add(date);
    }
    expandedDates = expandedDates; // Trigger reactivity
  }
</script>

<Card class="w-full max-w-4xl">
  <CardHeader>
    <div class="flex items-center justify-between">
      <CardTitle class="text-xl font-semibold flex items-center gap-2">
        ⏰ Evidence Timeline
        <Badge variant="outline">{timelineEvents.length} events</Badge>
      </CardTitle>
      <div class="flex gap-2">
        <Button variant="outline" size="sm">
          📊 Timeline Analysis  
        </Button>
        <Button variant="outline" size="sm">
          🗂️ Export Timeline
        </Button>
      </div>
    </div>
    {#if caseId}
      <p class="text-sm text-gray-600">Case: {caseId}</p>
    {/if}
  </CardHeader>

  <CardContent>
    {#if sortedEvents.length === 0}
      <div class="text-center py-8 text-gray-500">
        <div class="text-4xl mb-2">📅</div>
        <p class="text-lg font-medium mb-1">No timeline events</p>
        <p class="text-sm">Timeline events will appear here after evidence analysis</p>
      </div>
    {:else}
      <!-- Timeline -->
      <div class="relative">
        <!-- Timeline line -->
        <div class="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-300"></div>
        
        <div class="space-y-6">
          {#each Object.entries(groupedEvents) as [date, events]}
            <div class="relative">
              <!-- Date marker -->
              <div class="flex items-center gap-4 mb-4">
                <div class="relative">
                  <div class="w-4 h-4 bg-blue-500 rounded-full border-4 border-white shadow-md"></div>
                </div>
                <div class="flex-1">
                  <button 
                    class="text-left w-full flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                    onclick={() => toggleDate(date)}
                  >
                    <div>
                      <h3 class="font-semibold text-lg">{formatDate(date)}</h3>
                      <p class="text-sm text-gray-600">{events.length} event{events.length !== 1 ? 's' : ''}</p>
                    </div>
                    <div class="text-gray-400">
                      {expandedDates.has(date) ? '▼' : '▶'}
                    </div>
                  </button>
                </div>
              </div>

              <!-- Events for this date -->
              {#if expandedDates.has(date)}
                <div class="ml-12 space-y-3">
                  {#each events as event, index}
                    {@const categoryInfo = categoryConfig[event.category || 'discovery']}
                    <div class="bg-white border rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                      <!-- Event header -->
                      <div class="flex items-start justify-between mb-2">
                        <div class="flex items-center gap-3">
                          {#if event.time}
                            <div class="text-sm font-mono bg-gray-100 px-2 py-1 rounded">
                              {formatTime(event.time)}
                            </div>
                          {/if}
                          <Badge class="text-xs {categoryInfo.color}">
                            {categoryInfo.icon} {categoryInfo.label}
                          </Badge>
                        </div>
                        {#if event.confidence}
                          <div class="flex items-center gap-2">
                            <div class="text-xs text-gray-500">Confidence</div>
                            <div class="w-16 bg-gray-200 rounded-full h-1.5">
                              <div 
                                class="h-1.5 rounded-full {event.confidence > 0.8 ? 'bg-green-500' : 
                                                          event.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'}" 
                                style="width: {event.confidence * 100}%"
                              ></div>
                            </div>
                            <div class="text-xs text-gray-500">{Math.round(event.confidence * 100)}%</div>
                          </div>
                        {/if}
                      </div>

                      <!-- Event description -->
                      <p class="text-gray-800 mb-3 leading-relaxed">{event.event}</p>

                      <!-- Associated persons -->
                      {#if event.persons && event.persons.length > 0}
                        <div class="mb-3">
                          <h5 class="text-xs font-medium text-gray-600 mb-2">Persons Involved:</h5>
                          <div class="flex flex-wrap gap-1">
                            {#each event.persons as person}
                              <Badge variant="outline" class="text-xs bg-blue-50 text-blue-700 border-blue-200">
                                👤 {person}
                              </Badge>
                            {/each}
                          </div>
                        </div>
                      {/if}

                      <!-- Evidence source -->
                      {#if event.evidenceSource}
                        <div class="text-xs text-gray-500 bg-gray-50 px-2 py-1 rounded">
                          📄 Source: {event.evidenceSource}
                        </div>
                      {/if}
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
          {/each}
        </div>
      </div>

      <!-- Timeline actions -->
      <Separator class="my-6" />
      
      <div class="flex items-center justify-between">
        <div class="text-sm text-gray-600">
          {timelineEvents.length} events across {Object.keys(groupedEvents).length} day{Object.keys(groupedEvents).length !== 1 ? 's' : ''}
        </div>
        
        <div class="flex gap-2">
          <Button variant="outline" size="sm">
            🔍 Find Gaps
          </Button>
          <Button variant="outline" size="sm">
            🕸️ Show Connections
          </Button>
          <Button size="sm">
            📝 Generate Report
          </Button>
        </div>
      </div>
    {/if}
  </CardContent>
</Card>

<style>
  /* Timeline custom styles */
  .timeline-marker {
    position: absolute
    left: -6px;
    top: 12px;
    width: 12px;
    height: 12px;
    background: #3b82f6;
    border: 3px solid #ffffff;
    border-radius: 50%;
    box-shadow: 0 0 0 1px #e5e7eb;
  }
  
  .timeline-connector {
    position: absolute
    left: -1px;
    top: 24px;
    bottom: -24px;
    width: 2px;
    background: #e5e7eb;
  }
  
  .timeline-last .timeline-connector {
    display: none
  }
</style>
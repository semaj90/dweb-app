<!-- Legal AI Command Palette - Global Search Component -->
<script lang="ts">
  import { Command } from 'bits-ui';
  import { Search, FileText, Users, Calendar, Gavel } from 'lucide-svelte';
  import { cn } from '$lib/utils';
  import { createEventDispatcher } from 'svelte';

  // Exported props (use Svelte-style exports instead of $props/$bindable)
  let { open = $bindable() } = $props(); // boolean = false;
  let { onOpenChange = $bindable() } = $props(); // ((open: boolean) => void) | undefined;
  let { placeholder = $bindable() } = $props(); // string = 'Search cases, evidence, documents...';
  let { class = $bindable() } = $props(); // string = '';

  const dispatch = createEventDispatcher();

  // Mock data for legal AI platform
  const mockCommands = [
    {
      group: 'Cases',
      icon: Gavel,
      items: [
        { id: 'case-1', title: 'State v. Johnson', description: 'Active criminal case', keywords: ['criminal', 'theft', 'johnson'] },
        { id: 'case-2', title: 'Smith v. Corporation', description: 'Civil litigation', keywords: ['civil', 'corporate', 'smith'] },
        { id: 'case-3', title: 'People v. Williams', description: 'DUI case pending', keywords: ['dui', 'williams', 'traffic'] }
      ]
    },
    {
      group: 'Evidence',
      icon: FileText,
      items: [
        { id: 'evidence-1', title: 'Security Footage 2024-01-15', description: 'Video evidence', keywords: ['video', 'security', 'footage'] },
        { id: 'evidence-2', title: 'Financial Records', description: 'Bank statements', keywords: ['financial', 'bank', 'records'] },
        { id: 'evidence-3', title: 'Witness Statement - Martinez', description: 'Testimony transcript', keywords: ['witness', 'martinez', 'statement'] }
      ]
    },
    {
      group: 'People',
      icon: Users,
      items: [
        { id: 'person-1', title: 'John Smith', description: 'Defendant in case #2024-001', keywords: ['defendant', 'smith'] },
        { id: 'person-2', title: 'Detective Rodriguez', description: 'Lead investigator', keywords: ['detective', 'rodriguez', 'investigator'] },
        { id: 'person-3', title: 'Attorney Johnson', description: 'Prosecuting attorney', keywords: ['attorney', 'johnson', 'prosecutor'] }
      ]
    },
    {
      group: 'Documents',
      icon: FileText,
      items: [
        { id: 'doc-1', title: 'Motion to Dismiss', description: 'Filed 2024-01-20', keywords: ['motion', 'dismiss', 'filing'] },
        { id: 'doc-2', title: 'Search Warrant', description: 'Authorized 2024-01-18', keywords: ['warrant', 'search', 'authorized'] },
        { id: 'doc-3', title: 'Police Report', description: 'Initial incident report', keywords: ['police', 'report', 'incident'] }
      ]
    }
  ];

  function handleSelect(item: any) {
    dispatch('select', item);
    open = false;
  }

  function handleOpenChange(newOpen: boolean) {
    open = newOpen;
    onOpenChange?.(newOpen);
  }
</script>

<Command.Root
  bind:open
  openChange={handleOpenChange}
  class={cn(
    'legal-command-palette',
    'flex h-full w-full flex-col overflow-hidden rounded-md bg-popover text-popover-foreground',
    class
  )}
>
  <div class="flex items-center border-b px-3 legal-command-header">
    <Search class="mr-2 h-4 w-4 shrink-0 opacity-50" />
    <Command.Input
      placeholder={placeholder}
      class="flex h-11 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50 font-mono"
    />
  </div>

  <Command.List class="max-h-[300px] overflow-y-auto overflow-x-hidden legal-command-list">
    <Command.Empty class="py-6 text-center text-sm text-muted-foreground font-mono">
      No results found.
    </Command.Empty>

    {#each mockCommands as group}
      <Command.Group class="legal-command-group">
        <Command.GroupHeading class="px-2 py-1.5 text-xs font-medium text-muted-foreground font-mono uppercase tracking-wider flex items-center gap-2">
          <svelte:component this={group.icon} class="h-3 w-3" />
          {group.group}
        </Command.GroupHeading>

        {#each group.items as item}
          <Command.Item
            value={item.title + ' ' + item.description + ' ' + item.keywords.join(' ')}
            select={() => handleSelect(item)}
            class="relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none aria-selected:bg-accent aria-selected:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 legal-command-item font-mono"
          >
            <div class="flex items-start gap-3 w-full">
              <svelte:component this={group.icon} class="h-4 w-4 mt-0.5 text-muted-foreground flex-shrink-0" />
              <div class="flex flex-col gap-1 min-w-0 flex-1">
                <div class="font-medium text-sm">{item.title}</div>
                <div class="text-xs text-muted-foreground">{item.description}</div>
              </div>
            </div>
          </Command.Item>
        {/each}
      </Command.Group>
    {/each}
  </Command.List>
</Command.Root>

<style>
  /* Legal AI Command Palette Styling */
  :global(.legal-command-palette) {
    @apply bg-yorha-bg-primary border border-yorha-border shadow-xl;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
  }

  :global(.legal-command-header) {
    @apply border-yorha-border bg-yorha-bg-secondary;
  }

  :global(.legal-command-list) {
    @apply bg-yorha-bg-primary;
  }

  :global(.legal-command-group) {
    @apply border-yorha-border;
  }

  :global(.legal-command-item) {
    @apply hover:bg-yorha-bg-hover text-yorha-text-primary;
    @apply transition-colors duration-150;
  }

  :global(.legal-command-item[aria-selected="true"]) {
    @apply bg-yorha-accent text-yorha-text-accent;
  }
</style>
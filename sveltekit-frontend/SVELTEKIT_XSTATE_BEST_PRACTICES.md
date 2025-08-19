# 📚 SvelteKit 2 + XState + Loki.js Best Practices Guide

## 🏗️ Core SvelteKit 2 Best Practices

### 1. **No Side-Effects in `load` Functions**

`load` functions must be pure—do not write to stores or global state inside them. Always return data from `load` and pass it to components via props or `page.data`.

**❌ Bad:**

```typescript
// Don't do this in +page.ts
import { user } from "$lib/user";

export const load: PageLoad = async ({ fetch }) => {
  const response = await fetch("/api/user");
  // ❌ NEVER DO THIS!
  user.set(await response.json());
};
```

**✅ Good:**

```typescript
// Return data from load
export const load: PageLoad = async ({ fetch }) => {
  const response = await fetch("/api/user");
  return {
    user: await response.json(), // ✅ Return data
  };
};
```

### 2. **State & Stores with Context**

Use Svelte's `setContext`/`getContext` to share state down the component tree, not global modules.

**✅ Proper Context Usage:**

```typescript
// In +layout.svelte
import { setContext } from "svelte";
setContext("user", () => data.user);

// In child component
import { getContext } from "svelte";
const user = getContext("user");
```

### 3. **Component/Page State is Preserved**

Navigating between pages reuses components; lifecycle hooks like `onMount`/`onDestroy` do not rerun.

**Key Points:**

- Use `$derived` for reactive values that depend on `data`
- If you need to remount a component, use `{#key page.url.pathname}`
- State persists across navigation

**✅ Reactive Data Pattern:**

```typescript
<script>
  export let data;

  // ✅ Use $derived for reactive computations
  let wordCount = $derived(data.content.split(' ').length);
  let estimatedReadingTime = $derived(wordCount / 250);
</script>
```

### 4. **Store State in the URL for Persistence**

Use URL search params for state that should persist across reloads or affect SSR.

**✅ URL State Pattern:**

```typescript
// Store filters in URL
const url = new URL(window.location);
url.searchParams.set("filter", "active");
goto(url.toString());
```

### 5. **Ephemeral State in Snapshots**

Use SvelteKit snapshots for UI state that should persist across navigation but not reloads.

---

## 🎯 XState Integration Best Practices

### 6. **State Machine Architecture**

Use XState machines for complex state management, especially for multi-step processes and data management.

**✅ Machine-First Approach:**

```typescript
// Create machines for complex state
import { useMachine } from "@xstate/svelte";
import { caseManagementMachine } from "$lib/machines/caseManagementMachine";

export let data;

const { state, send } = useMachine(
  caseManagementMachine.withContext({
    ...caseManagementMachine.context,
    cases: data.initialCases, // ✅ Use SSR data to hydrate machine
  }),
);
```

### 7. **SSR-Compatible Machine Hydration**

Always hydrate XState machines with SSR data to prevent hydration mismatches.

**✅ SSR Hydration Pattern:**

```typescript
// +page.server.ts
export const load: PageServerLoad = async () => {
  return {
    initialCases: await getCases(),
    initialFilters: getDefaultFilters(),
  };
};

// +page.svelte
const machineWithInitialData = caseManagementMachine.withContext({
  ...caseManagementMachine.context,
  cases: data.initialCases,
  filters: data.initialFilters,
});
```

### 8. **Machine State Reactivity**

Use reactive statements to extract machine state for UI rendering.

**✅ Reactive Machine State:**

```typescript
const { state, send } = useMachine(myMachine);

// ✅ Extract reactive values
$: currentState = $state.value;
$: contextData = $state.context;
$: isLoading = $state.matches("loading");
$: error = $state.context.error;
```

### 9. **Event-Driven Updates**

Use machine events for all state changes instead of direct mutations.

**✅ Event-Driven Pattern:**

```typescript
// ✅ Send events to machine
function handleSearch(query: string) {
  send({ type: "SEARCH", query });
}

function handleFilter(filter: FilterType) {
  send({ type: "APPLY_FILTER", filter });
}
```

---

## 🗄️ Loki.js Caching Best Practices

### 10. **Cache-First Data Loading**

Always check cache before making API requests.

**✅ Cache-First Pattern:**

```typescript
// In machine actors
searchCases: fromPromise(async ({ input }) => {
  // ✅ Try cache first
  const cached = caseCacheManager.get();
  if (cached.length > 0 && isCacheValid()) {
    return { data: cached, fromCache: true };
  }

  // ✅ Fallback to API
  const response = await fetch('/api/cases');
  const data = await response.json();

  // ✅ Update cache
  caseCacheManager.upsert(data);
  return { data, fromCache: false };
}),
```

### 11. **Smart Cache Invalidation**

Implement proper cache invalidation strategies.

**✅ Cache Invalidation Pattern:**

```typescript
// Invalidate cache on mutations
updateCase: fromPromise(async ({ input }) => {
  const response = await fetch(`/api/cases/${input.id}`, {
    method: 'PUT',
    body: JSON.stringify(input.data),
  });

  const updatedCase = await response.json();

  // ✅ Update cache with new data
  caseCacheManager.upsert([updatedCase]);

  return updatedCase;
}),
```

### 12. **TTL-Based Cache Strategy**

Set appropriate TTL (Time To Live) for different data types.

**✅ TTL Configuration:**

```typescript
// Different TTL for different data types
export const cacheConfig = {
  cases: { ttl: 10 * 60 * 1000 }, // 10 minutes
  evidence: { ttl: 15 * 60 * 1000 }, // 15 minutes
  users: { ttl: 60 * 60 * 1000 }, // 1 hour
  staticData: { ttl: 24 * 60 * 60 * 1000 }, // 24 hours
};
```

---

## 🔄 Real-time Integration Best Practices

### 13. **WebSocket State Management**

Use XState to manage WebSocket connection states.

**✅ WebSocket Machine Pattern:**

```typescript
const { state, send } = useMachine(realtimeMachine);

$: isConnected = $state.context.isConnected;
$: unreadCount = $state.context.unreadCount;

// ✅ Handle connection lifecycle
onMount(() => {
  if (user) {
    send({ type: "CONNECT", userId: user.id });
  }
});
```

### 14. **Cache Synchronization**

Keep cache in sync with real-time updates.

**✅ Real-time Cache Sync:**

```typescript
// Update cache when receiving real-time updates
processRealtimeUpdate: assign({
  recentUpdates: ({ context, event }) => {
    if (event.type === 'SOCKET_MESSAGE') {
      // ✅ Update cache based on update type
      updateCacheFromRealtimeData(event.data);

      return [newUpdate, ...context.recentUpdates];
    }
    return context.recentUpdates;
  },
}),
```

---

## 🧪 Development & Debugging Best Practices

### 15. **XState Inspector Usage**

Always use XState Inspector during development.

**✅ Development Setup:**

```typescript
// src/lib/config/xstate-dev.ts
import { createBrowserInspector } from "@xstate/inspect";

if (import.meta.env.DEV) {
  createBrowserInspector({
    iframe: false, // Use popup window
    url: "https://stately.ai/viz?inspect",
  });
}
```

### 16. **Cache Debugging**

Log cache operations during development.

**✅ Cache Debugging:**

```typescript
// Enable cache debugging in development
const DEBUG_CACHE = import.meta.env.DEV;

if (DEBUG_CACHE) {
  console.log("Cache hit:", cachedData.length);
  console.log("Cache stats:", cacheManager.getStats());
}
```

### 17. **Machine Testing Patterns**

Write tests for machine logic, not just components.

**✅ Machine Testing:**

```typescript
// Test machine states and transitions
import { describe, it, expect } from "vitest";
import { caseManagementMachine } from "$lib/machines/caseManagementMachine";

describe("Case Management Machine", () => {
  it("should transition to searching when SEARCH event is sent", () => {
    const nextState = caseManagementMachine.transition("idle", {
      type: "SEARCH",
      query: "test",
    });
    expect(nextState.value).toBe("searching");
  });
});
```

---

## 🚀 Performance Best Practices

### 18. **Lazy Loading Machines**

Load state machines only when needed.

**✅ Lazy Machine Loading:**

```typescript
// Load machines on demand
const loadCaseMachine = () => import("$lib/machines/caseManagementMachine");
const loadAIMachine = () => import("$lib/machines/aiAssistantMachine");

// Use in components
onMount(async () => {
  const { caseManagementMachine } = await loadCaseMachine();
  // Initialize machine
});
```

### 19. **Memory Management**

Monitor and control memory usage.

**✅ Memory Management:**

```typescript
// Set cache limits
export const cacheManager = new CacheManager(collection, {
  ttl: 10 * 60 * 1000, // 10 minutes
  maxSize: 1000, // Max 1000 items
  cleanupInterval: 5000, // Cleanup every 5 seconds
});

// Regular cleanup
setInterval(() => {
  cacheManager.cleanup();
}, cacheManager.cleanupInterval);
```

### 20. **Bundle Optimization**

Optimize bundle size for XState and Loki.js.

**✅ Bundle Optimization:**

```typescript
// vite.config.ts
export default defineConfig({
  optimizeDeps: {
    include: ["lokijs", "xstate", "@xstate/svelte"],
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          "state-machines": ["xstate", "@xstate/svelte"],
          cache: ["lokijs"],
        },
      },
    },
  },
});
```

---

## 🏭 Production Best Practices

### 21. **Environment-Specific Configuration**

Configure differently for development vs production.

**✅ Environment Configuration:**

```typescript
// Production settings
const config = {
  xstate: {
    devTools: import.meta.env.DEV,
    inspect: import.meta.env.DEV,
  },
  cache: {
    debug: import.meta.env.DEV,
    persist: !import.meta.env.DEV, // Only persist in production
  },
  websocket: {
    url: import.meta.env.VITE_WS_URL || "ws://localhost:3001",
    reconnectAttempts: import.meta.env.PROD ? 10 : 3,
  },
};
```

### 22. **Error Boundaries**

Implement proper error handling for machines.

**✅ Error Handling:**

```typescript
// Machine with error handling
states: {
  loading: {
    invoke: {
      src: 'loadData',
      onDone: {
        target: 'success',
        actions: 'setData',
      },
      onError: {
        target: 'error',
        actions: 'setError',
      },
    },
  },
  error: {
    on: {
      RETRY: {
        target: 'loading',
        guard: 'canRetry', // ✅ Limit retry attempts
      },
      CLEAR_ERROR: {
        target: 'idle',
        actions: 'clearError',
      },
    },
  },
},
```

### 23. **Monitoring & Analytics**

Track machine state transitions and performance.

**✅ Monitoring Pattern:**

```typescript
// Track state transitions
const { state, send } = useMachine(myMachine, {
  logger: (log) => {
    if (import.meta.env.PROD) {
      // Send to analytics
      analytics.track("state_transition", {
        machine: log.machine.id,
        from: log.state.value,
        event: log.event.type,
      });
    }
  },
});
```

---

## 🎨 Svelte 5 + Bits UI v2 + Melt UI Integration Best Practices

### 24. **Component Composition with mergeProps**

Use `mergeProps` from Bits UI for proper component composition and prop forwarding.

**✅ Bits UI v2 Component Pattern:**

```typescript
<script lang="ts">
  import { mergeProps } from 'bits-ui';
  import { Button as ButtonPrimitive } from 'bits-ui';
  
  interface Props {
    variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
    size?: 'default' | 'sm' | 'lg' | 'icon';
    class?: string;
    children?: import('svelte').Snippet;
  }
  
  let {
    variant = 'default',
    size = 'default', 
    class: className = '',
    children,
    ...restProps
  }: Props = $props();
  
  // ✅ Merge props with Bits UI primitives
  const buttonProps = mergeProps(restProps, {
    class: cn(buttonVariants({ variant, size }), className)
  });
</script>

<ButtonPrimitive.Root {...buttonProps}>
  {#if children}
    {@render children()}
  {/if}
</ButtonPrimitive.Root>
```

### 25. **Snippet-based Slot Forwarding**

Use Svelte 5 snippets for flexible component composition with Bits UI.

**✅ Snippet Composition Pattern:**

```typescript
<script lang="ts">
  import { Dialog as DialogPrimitive } from 'bits-ui';
  
  interface Props {
    title?: string;
    description?: string;
    children: import('svelte').Snippet;
    footer?: import('svelte').Snippet<[{ close: () => void }]>;
  }
  
  let { title, description, children, footer }: Props = $props();
</script>

<DialogPrimitive.Root>
  <DialogPrimitive.Content>
    {#if title}
      <DialogPrimitive.Title>{title}</DialogPrimitive.Title>
    {/if}
    {#if description}
      <DialogPrimitive.Description>{description}</DialogPrimitive.Description>
    {/if}
    
    <!-- ✅ Render child snippet -->
    {@render children()}
    
    {#if footer}
      <!-- ✅ Pass context to footer snippet -->
      {@render footer({ close: () => {} })}
    {/if}
  </DialogPrimitive.Content>
</DialogPrimitive.Root>
```

### 26. **Prop Consolidation for Duplicate Variables**

Consolidate duplicate prop declarations to fix syntax errors.

**❌ Duplicate Props (causes errors):**

```typescript
<script lang="ts">
  let { size = 'md' } = $props();
  // ... other code ...
  let { size = $bindable() } = $props(); // ❌ Duplicate declaration
</script>
```

**✅ Consolidated Props Pattern:**

```typescript
<script lang="ts">
  interface Props {
    size?: 'sm' | 'md' | 'lg' | 'xl';
    color?: 'blue' | 'green' | 'red' | 'yellow' | 'gray' | 'white';
    label?: string;
    inline?: boolean;
  }
  
  let {
    size = 'md',
    color = 'blue', 
    label = 'Loading...',
    inline = false
  }: Props = $props();
  
  // ✅ Single prop destructuring with proper types
</script>
```

### 27. **Melt UI Integration with Svelte 5**

Use Melt UI builders and components with Svelte 5 runes system.

**✅ Melt UI Builder Pattern (Reactive Getters/Setters):**

```typescript
<script lang="ts">
  import { Toggle } from "melt/builders";
  
  interface Props {
    value?: boolean;
    onValueChange?: (value: boolean) => void;
  }
  
  let { value = false, onValueChange }: Props = $props();
  
  // ✅ Use Melt UI builder with Svelte 5 runes
  const toggle = new Toggle({
    value: () => value,
    onValueChange: (v) => {
      value = v;
      onValueChange?.(v);
    }
  });
</script>

<button {...toggle.trigger}>
  {toggle.value ? "On" : "Off"}
</button>
```

**✅ Melt UI Component Pattern (Traditional Approach):**

```typescript
<script lang="ts">
  import { Toggle } from "melt/components";
  
  interface Props {
    initialValue?: boolean;
    onValueChange?: (value: boolean) => void;
  }
  
  let { initialValue = false, onValueChange }: Props = $props();
  let value = $state(initialValue);
  
  // ✅ React to value changes
  $effect(() => {
    onValueChange?.(value);
  });
</script>

<Toggle bind:value>
  {#snippet children(toggle)}
    <button {...toggle.trigger} class="toggle-button">
      {toggle.value ? "✅ On" : "❌ Off"}
    </button>
  {/snippet}
</Toggle>
```

**✅ Complex Melt UI Integration (Dialog Example):**

```typescript
<script lang="ts">
  import { Dialog } from "melt/components";
  
  interface Props {
    open?: boolean;
    title: string;
    description?: string;
    children: import('svelte').Snippet;
    onOpenChange?: (open: boolean) => void;
  }
  
  let { 
    open = false, 
    title, 
    description, 
    children, 
    onOpenChange 
  }: Props = $props();
  
  // ✅ Use $effect for prop synchronization
  $effect(() => {
    onOpenChange?.(open);
  });
</script>

<Dialog bind:open closeOnOutsideClick>
  {#snippet children(dialog)}
    <button {...dialog.trigger} class="dialog-trigger">
      Open Dialog
    </button>
    
    {#if dialog.open}
      <div {...dialog.overlay} class="dialog-overlay">
        <div {...dialog.content} class="dialog-content">
          <h2 {...dialog.title}>{title}</h2>
          {#if description}
            <p {...dialog.description}>{description}</p>
          {/if}
          
          {@render children()}
          
          <button {...dialog.close} class="dialog-close">
            Close
          </button>
        </div>
      </div>
    {/if}
  {/snippet}
</Dialog>
```

### 28. **Event Handler Chaining**

Properly chain event handlers when using Bits UI + custom logic.

**✅ Event Handler Chaining:**

```typescript
<script lang="ts">
  import { Button as ButtonPrimitive } from 'bits-ui';
  
  interface Props {
    onclick?: (event: MouseEvent) => void;
    variant?: 'default' | 'destructive';
    children: import('svelte').Snippet;
  }
  
  let { onclick, variant = 'default', children, ...restProps }: Props = $props();
  
  // ✅ Chain custom handler with primitive handler
  function handleClick(event: MouseEvent) {
    // Custom logic first
    console.log('Button clicked:', variant);
    
    // Then call user-provided handler
    onclick?.(event);
  }
</script>

<ButtonPrimitive.Root 
  {...restProps}
  onclick={handleClick}
  class={cn(buttonVariants({ variant }))}
>
  {@render children()}
</ButtonPrimitive.Root>
```

### 29. **Class Merging with clsx**

Use `clsx` for proper class merging in component libraries.

**✅ Class Merging Pattern:**

```typescript
<script lang="ts">
  import { clsx, type ClassValue } from 'clsx';
  import { twMerge } from 'tailwind-merge';
  
  // ✅ Utility for class merging
  export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
  }
  
  interface Props {
    class?: string;
    variant?: 'default' | 'secondary' | 'destructive';
    size?: 'sm' | 'md' | 'lg';
  }
  
  let { 
    class: className = '',
    variant = 'default',
    size = 'md',
    ...restProps 
  }: Props = $props();
  
  // ✅ Define variant styles
  const variants = {
    default: 'bg-primary text-primary-foreground hover:bg-primary/90',
    secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
    destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90'
  };
  
  const sizes = {
    sm: 'h-9 rounded-md px-3',
    md: 'h-10 px-4 py-2', 
    lg: 'h-11 rounded-md px-8'
  };
  
  // ✅ Merge all classes properly
  const buttonClass = cn(
    'inline-flex items-center justify-center rounded-md font-medium transition-colors',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
    'disabled:pointer-events-none disabled:opacity-50',
    variants[variant],
    sizes[size],
    className
  );
</script>
```

### 30. **Accessibility with Bits UI Primitives**

Leverage Bits UI's built-in accessibility features properly.

**✅ Accessible Component Pattern:**

```typescript
<script lang="ts">
  import { 
    Select as SelectPrimitive,
    Label as LabelPrimitive 
  } from 'bits-ui';
  
  interface Props {
    label: string;
    placeholder?: string;
    value?: string;
    onValueChange?: (value: string) => void;
    items: Array<{ value: string; label: string }>;
    disabled?: boolean;
    required?: boolean;
  }
  
  let {
    label,
    placeholder = 'Select an option...',
    value = '',
    onValueChange,
    items,
    disabled = false,
    required = false
  }: Props = $props();
</script>

<!-- ✅ Accessible select with proper labeling -->
<LabelPrimitive.Root class="select-label">
  {label}
  {#if required}
    <span class="text-destructive" aria-label="required">*</span>
  {/if}
</LabelPrimitive.Root>

<SelectPrimitive.Root 
  {value} 
  onValueChange={onValueChange}
  {disabled}
  {required}
>
  <SelectPrimitive.Trigger 
    class={cn(
      'flex h-10 w-full items-center justify-between rounded-md border px-3 py-2',
      disabled && 'cursor-not-allowed opacity-50'
    )}
    aria-label={label}
  >
    <SelectPrimitive.Value {placeholder} />
    <SelectPrimitive.Icon />
  </SelectPrimitive.Trigger>
  
  <SelectPrimitive.Content>
    {#each items as item}
      <SelectPrimitive.Item value={item.value}>
        <SelectPrimitive.ItemText>{item.label}</SelectPrimitive.ItemText>
      </SelectPrimitive.Item>
    {/each}
  </SelectPrimitive.Content>
</SelectPrimitive.Root>
```

---

## 📋 Key Principles Summary

### **Never Do This:**

- ❌ Mutate global state in `load` functions
- ❌ Use global stores for data that should use context
- ❌ Forget to use `$derived` for reactive values in preserved components
- ❌ Store persistent state in ephemeral places
- ❌ Create machines without SSR hydration
- ❌ Make API calls without checking cache first
- ❌ Enable XState DevTools in production
- ❌ Ignore cache TTL and memory limits

### **Always Do This:**

- ✅ Return data from `load` functions
- ✅ Use context for component tree state sharing
- ✅ Use `$derived` for reactive computations
- ✅ Store persistent state in URL params
- ✅ Hydrate machines with SSR data
- ✅ Check cache before API calls
- ✅ Use machines for complex state management
- ✅ Implement proper error boundaries
- ✅ Monitor performance and state transitions
- ✅ Test machine logic independently

### **Context-Aware Development:**

- 🧠 **SSR First**: Always consider server-side rendering
- 🔄 **State Preservation**: Components persist across navigation
- 📱 **Progressive Enhancement**: Start with SSR, enhance with interactivity
- ⚡ **Performance**: Cache strategically, load lazily
- 🐛 **Debugging**: Use visual tools for state management
- 🏭 **Production Ready**: Monitor, measure, and optimize

This comprehensive guide ensures your SvelteKit + XState + Loki.js application follows best practices for performance, maintainability, and user experience! 🚀

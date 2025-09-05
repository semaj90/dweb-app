<script lang="ts">
  interface Props {
    open?: any;
  }
  let { open = false } = $props();

  import { page } from "$app/stores";
  import Button from "$lib/components/ui/Button.svelte";
  import { cn } from "$lib/utils";
  import {
    BarChart3,
    Bot,
    Briefcase,
    ChevronRight,
    FileBarChart,
    FileText,
    Home,
    Layers,
    Plus,
    Scale,
    Search,
    Settings,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  let mounted = false;

  onMount(() => {
    mounted = true;
  });

  let currentPath = $derived($page.url.pathname)

  const navigation = [
    {
      name: "Dashboard",
      href: "/",
      icon: Home,
      current: currentPath === "/",
    },
    {
      name: "Evidence",
      href: "/evidence",
      icon: FileText,
      current: currentPath.startsWith("/evidence"),
      badge: "12 New",
    },
    {
      name: "Cases",
      href: "/cases",
      icon: Briefcase,
      current: currentPath.startsWith("/cases"),
    },
    {
      name: "AI Assistant",
      href: "/ai-assistant",
      icon: Bot,
      current: currentPath.startsWith("/ai-assistant"),
      badge: "Beta",
    },
    {
      name: "Evidence Canvas",
      href: "/interactive-canvas",
      icon: Layers,
      current: currentPath.startsWith("/interactive-canvas"),
    },
  ];

  const analytics = [
    {
      name: "Analytics",
      href: "/analytics",
      icon: BarChart3,
      current: currentPath.startsWith("/analytics"),
    },
    {
      name: "Reports",
      href: "/reports",
      icon: FileBarChart,
      current: currentPath.startsWith("/reports"),
    },
  ];

  const settings = [
    {
      name: "Settings",
      href: "/settings",
      icon: Settings,
      current: currentPath.startsWith("/settings"),
    },
  ];

  function closeSidebar() {
    open = false;
  }
</script>

<!-- Mobile backdrop -->
{#if open}
  <div
    class="fixed inset-0 z-40 bg-black/50 lg:hidden"
    on:click={closeSidebar}
    role="button"
    tabindex="0"
    on:keydown={(e) => e.key === "Enter" && closeSidebar()}
  ></div>
{/if}

<!-- Sidebar -->
<aside
  class={cn(
    "fixed top-0 left-0 z-50 h-full w-64 transform bg-nier-surface border-r border-nier-gray transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0",
    open ? "translate-x-0" : "-translate-x-full"
  )}
>
  <div class="flex h-full flex-col">
    <!-- Logo section -->
    <div class="flex h-16 items-center border-b border-nier-gray px-6">
      <div class="flex items-center gap-3">
        <div
          class="w-8 h-8 bg-crimson-gradient rounded-md flex items-center justify-center nier-glow"
        >
          <Scale class="h-5 w-5 text-white" />
        </div>
        <div>
          <h1 class="text-sm font-semibold text-foreground">
            Enhanced Legal AI
          </h1>
          <p class="text-xs text-muted-foreground">
            Justice Through Technology
          </p>
        </div>
      </div>
    </div>

    <!-- Quick actions -->
    <div class="p-4 border-b border-nier-gray">
      <div class="grid grid-cols-2 gap-2">
        <Button variant="primary" size="sm" class="justify-start">
          <Plus class="mr-2 h-4 w-4" />
          New Case
        </Button>
        <Button variant="ghost" size="sm" class="justify-start">
          <Search class="mr-2 h-4 w-4" />
          Search
        </Button>
      </div>
    </div>

    <!-- Navigation -->
    <nav class="flex-1 overflow-y-auto py-4">
      <div class="px-3 space-y-1">
        <!-- Main navigation -->
        <div class="space-y-1">
          {#each navigation as item}
            <a
              href={item.href}
              class={cn(
                "group flex items-center justify-between px-3 py-2 text-sm font-medium rounded-md transition-all duration-200",
                item.current
                  ? "bg-harvard-crimson text-white shadow-nier-glow"
                  : "text-muted-foreground hover:text-foreground hover:bg-nier-surface-light"
              )}
              on:click={closeSidebar}
            >
              <div class="flex items-center">
                {@const Icon = item.icon}
                <Icon
                  class={cn(
                    "mr-3 h-5 w-5 flex-shrink-0",
                    item.current
                      ? "text-white"
                      : "text-muted-foreground group-hover:text-foreground"
                  )}
                />
                {item.name}
              </div>

              {#if item.badge}
                <span
                  class={cn(
                    "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium",
                    item.current
                      ? "bg-white/20 text-white"
                      : "bg-harvard-crimson text-white"
                  )}
                >
                  {item.badge}
                </span>
              {/if}

              {#if item.current}
                <ChevronRight class="h-4 w-4 text-white" />
              {/if}
            </a>
          {/each}
        </div>

        <!-- Analytics section -->
        <div class="pt-4">
          <h3
            class="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider"
          >
            Analytics
          </h3>
          <div class="mt-2 space-y-1">
            {#each analytics as item}
              <a
                href={item.href}
                class={cn(
                  "group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-all duration-200",
                  item.current
                    ? "bg-harvard-crimson text-white shadow-nier-glow"
                    : "text-muted-foreground hover:text-foreground hover:bg-nier-surface-light"
                )}
                on:click={closeSidebar}
              >
                {@const Icon = item.icon}
                <Icon
                  class={cn(
                    "mr-3 h-5 w-5 flex-shrink-0",
                    item.current
                      ? "text-white"
                      : "text-muted-foreground group-hover:text-foreground"
                  )}
                />
                {item.name}
              </a>
            {/each}
          </div>
        </div>

        <!-- Settings section -->
        <div class="pt-4">
          <h3
            class="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider"
          >
            System
          </h3>
          <div class="mt-2 space-y-1">
            {#each settings as item}
              <a
                href={item.href}
                class={cn(
                  "group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-all duration-200",
                  item.current
                    ? "bg-harvard-crimson text-white shadow-nier-glow"
                    : "text-muted-foreground hover:text-foreground hover:bg-nier-surface-light"
                )}
                on:click={closeSidebar}
              >
                {@const Icon = item.icon}
                <Icon
                  class={cn(
                    "mr-3 h-5 w-5 flex-shrink-0",
                    item.current
                      ? "text-white"
                      : "text-muted-foreground group-hover:text-foreground"
                  )}
                />
                {item.name}
              </a>
            {/each}
          </div>
        </div>
      </div>
    </nav>

    <!-- Status indicator -->
    <div class="p-4 border-t border-nier-gray">
      <div class="flex items-center gap-3 p-3 bg-nier-surface-light rounded-md">
        <div
          class="w-3 h-3 bg-legal-success rounded-full animate-nier-pulse"
        ></div>
        <div class="flex-1 min-w-0">
          <p class="text-sm font-medium text-foreground">System Status</p>
          <p class="text-xs text-muted-foreground truncate">
            All systems operational
          </p>
        </div>
      </div>
    </div>
  </div>
</aside>

<style>
  /* @unocss-include */
  /* Custom styles for Nier aesthetic */
  aside {
    background: linear-gradient(
      180deg,
      var(--color-ui-surface) 0%,
      var(--color-primary-dark-gray) 100%
    );
  }
  .nier-glow {
    box-shadow: 0 0 10px rgba(165, 28, 48, 0.3);
  }
  /* Enhance scrollbar for sidebar */
  nav::-webkit-scrollbar {
    width: 4px;
  }
  nav::-webkit-scrollbar-track {
    background: transparent
  }
  nav::-webkit-scrollbar-thumb {
    background: var(--color-accent-crimson);
    border-radius: 2px;
  }
</style>

<script lang="ts">
  // Sidebar navigation logic will go here
  // For now, a simple list of items
  const navItems = [
    { name: 'Dashboard', icon: '📊', href: '/dashboard' },
    { name: 'Cases', icon: '🗂️', href: '/cases' },
    { name: 'Criminals', icon: '🧑‍⚖️', href: '/criminals' },
    { name: 'Statutes', icon: '⚖️', href: '/statutes' },
    { name: 'Interactive Canvas', icon: '🎨', href: '/interactive-canvas' },
    { name: 'About', icon: 'ℹ️', href: '/about' },
  ];

  let activeItem = 'Interactive Canvas'; // Default active item for the new UI
</script>

<aside class="menu min-h-full w-80 bg-base-200 p-4 text-base-content">
  <h2 class="menu-title">Menu</h2>
  <ul class="space-y-2">
    {#each navItems as item}
      <li>
        <a
          href={item.href}
          class="flex items-center gap-2"
          class:active={item.name === activeItem}
          aria-current={item.name === activeItem ? 'page' : undefined}
        >
          <span class="text-xl">{item.icon}</span>
          {item.name}
        </a>
      </li>
    {/each}
  </ul>
</aside>
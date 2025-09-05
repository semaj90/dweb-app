<script lang="ts">
  import { goto } from '$app/navigation';
  import SearchBar from '$lib/components/+SearchBar.svelte';

  export let user: any | undefined; // Made user prop optional
  export let title: string = 'WardenNet'; // Default title

  // Svelte action for SPA navigation
  function navigate(node: HTMLAnchorElement) {
    node.addEventListener('click', (e) => {
      if (!e.ctrlKey && !e.metaKey && !e.shiftKey && !e.altKey && e.button === 0) {
        e.preventDefault();
        const href = node.getAttribute('href');
        if (href) {
          goto(href);
        }
      }
    });
    return {
      destroy() {
        node.removeEventListener('click', () => {});
      }
    };
  }
</script>

<nav class="navbar bg-base-100 shadow-sm">
  <div class="flex-none lg:hidden">
    <label for="my-drawer-2" class="btn btn-square btn-ghost">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        class="inline-block h-5 w-5 stroke-current"
        ><path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M4 6h16M4 12h16M4 18h16"
        ></path></svg
      >
    </label>
  </div>
  <div class="flex-1">
    <a class="btn btn-ghost text-xl" href="/">{title}</a>
  </div>
  <div class="flex-none gap-2">
    <SearchBar />
    {#if user}
      <div class="dropdown dropdown-end">
        <div tabindex="0" role="button" class="avatar btn btn-ghost">
          {#if user.image}
            <div class="w-10 rounded-full">
              <img alt="Profile" src={user.image} />
            </div>
          {:else}
            <div class="placeholder avatar">
              <div class="w-10 rounded-full bg-neutral text-neutral-content">
                <span class="text-xl">{user.name ? user.name[0].toUpperCase() : user.email[0].toUpperCase()}</span>
              </div>
            </div>
          {/if}
        </div>
        <ul
          tabindex="0"
          class="menu dropdown-content menu-sm z-[1] mt-3 w-52 rounded-box bg-base-100 p-2 shadow"
        >
          <li><a href="/profile">Profile</a></li>
          <li>
            <form action="/logout" method="POST">
              <button type="submit">Logout</button>
            </form>
          </li>
        </ul>
      </div>
    {:else}
      <a href="/login" class="btn btn-primary" use:navigate>Sign In</a>
      <a href="/register" class="btn btn-secondary" use:navigate>Register</a>
    {/if}
  </div>
</nav>
